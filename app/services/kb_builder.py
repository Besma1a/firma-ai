"""
Knowledge-base builder
======================
Reads the structured crops JSON and writes one focused markdown chunk per
(crop x topic x language). These chunks are what the embedding model later
turns into vectors, so the *quality* of these chunks = the quality of your
chatbot.

This module is import-safe: nothing runs at import time. Call build_kb().
"""
import json
from pathlib import Path

from app.config import settings


# --- Localized labels so chunks read naturally in their language ---
LABELS = {
    "fr": {"planting": "Calendrier de plantation", "water": "Besoins en eau",
           "diseases": "Maladies courantes", "yield": "Rendement et profil",
           "fertilization": "Fertilisation NPK", "treatment": "Traitement"},
    "ar": {"planting": "تقويم الزراعة", "water": "الاحتياجات المائية",
           "diseases": "الأمراض الشائعة", "yield": "الإنتاج والمعلومات",
           "fertilization": "التسميد NPK", "treatment": "العلاج"},
    "en": {"planting": "Planting calendar", "water": "Water requirements",
           "diseases": "Common diseases", "yield": "Yield and profile",
           "fertilization": "NPK fertilization", "treatment": "Treatment"},
}

# -----------------------------------------------------------------------
# Treatment data: (crop_id, disease_id) → {lang: (active, preventive, curative, cultural)}
# Tuple order: (active_ingredients, preventive, curative, cultural/biological)
# -----------------------------------------------------------------------
TREATMENTS = {
    # ==== TOMATO ====
    ("tomato", "late_blight"): {
        "en": ("mancozeb 80% WP, copper oxychloride 50% WP",
               "Mancozeb 80% WP at 2.5 kg/ha weekly during humid periods.",
               "Copper oxychloride 50% WP at 3 kg/ha at first symptoms. Remove infected plants.",
               "Destroy infected plant debris. Improve airflow between rows."),
        "fr": ("mancozèbe 80% WP, oxychlorure de cuivre 50% WP",
               "Mancozèbe 80% WP à 2.5 kg/ha en préventif, hebdomadaire en périodes humides.",
               "Oxychlorure de cuivre 50% WP à 3 kg/ha dès les premiers symptômes. Arracher les plantes malades.",
               "Éliminer les débris végétaux infectés. Améliorer la ventilation entre les rangs."),
        "ar": ("مانكوزيب 80% WP، أوكسيكلوريد النحاس 50% WP",
               "رش مانكوزيب 80% WP بـ2.5 كغ/هكتار وقائيا، أسبوعيا في فترات الرطوبة.",
               "عند ظهور الأعراض: أوكسيكلوريد النحاس 50% WP بـ3 كغ/هكتار. إزالة النباتات المصابة.",
               "إتلاف مخلفات النباتات. تحسين التهوية بين الصفوف."),
    },
    ("tomato", "early_blight"): {
        "en": ("chlorothalonil 75% WP",
               "Chlorothalonil 75% WP at 2 kg/ha every 7–10 days. Mulching reduces splash infection.",
               "Increase spray frequency to every 7 days under heavy pressure.",
               "Crop rotation minimum 3 years. Remove lower infected leaves promptly."),
        "fr": ("chlorothalonil 75% WP",
               "Chlorothalonil 75% WP à 2 kg/ha toutes les 7–10 jours. Le paillage réduit les éclaboussures.",
               "Raccourcir les intervalles à 7 jours en cas de forte pression.",
               "Rotation culturale minimale de 3 ans. Supprimer les feuilles basses infectées."),
        "ar": ("كلوروثالونيل 75% WP",
               "كلوروثالونيل 75% WP بـ2 كغ/هكتار كل 7-10 أيام. التغطية بالمهاد تقلل العدوى برذاذ الماء.",
               "تقصير الفترة إلى 7 أيام عند الضغط الشديد.",
               "دوران محصولي 3 سنوات على الأقل. إزالة الأوراق السفلية المصابة."),
    },
    ("tomato", "tylcv"): {
        "en": ("imidacloprid 200 SL (for whitefly vector)",
               "No chemical cure for the virus. Use imidacloprid 200 SL at 0.5 L/ha to control whitefly vectors.",
               "Remove and burn infected plants. Insect-proof netting in nursery stage.",
               "Plant resistant hybrid varieties. Yellow sticky traps to monitor whitefly populations."),
        "fr": ("imidaclopride 200 SL (contre les aleurodes vecteurs)",
               "Aucun traitement chimique contre le virus. Imidaclopride 200 SL à 0.5 L/ha contre les aleurodes vecteurs.",
               "Arracher et brûler les plantes atteintes. Filets anti-insectes en pépinière.",
               "Utiliser des variétés hybrides résistantes. Pièges jaunes englués pour surveiller les aleurodes."),
        "ar": ("إيميداكلوبريد 200 SL (ضد ذبابة الورقة الناقلة)",
               "لا علاج كيميائي للفيروس. إيميداكلوبريد 200 SL بـ0.5 ل/هكتار للقضاء على ذبابة الورقة الناقلة.",
               "اقتلاع وحرق النباتات المصابة. شبكات مضادة للحشرات في مرحلة المشتل.",
               "زراعة أصناف هجينة مقاومة. مصائد صفراء لمتابعة أعداد الذبابة البيضاء."),
    },
    ("tomato", "fusarium_wilt"): {
        "en": ("no effective chemical cure",
               "No chemical treatment effective. Use resistant varieties with 'VFN' suffix.",
               "Soil solarization for 6 weeks during summer (plastic mulch on moist soil).",
               "Long crop rotation of 4 years minimum. Avoid replanting in infested fields."),
        "fr": ("aucun traitement chimique efficace",
               "Aucun traitement chimique efficace. Utiliser des variétés résistantes (suffixe « VFN »).",
               "Solarisation du sol 6 semaines en été (bâche plastique sur sol humide).",
               "Rotation longue de 4 ans minimum. Éviter la replantation sur parcelles contaminées."),
        "ar": ("لا علاج كيميائي فعّال",
               "لا علاج كيميائي مجدٍ. استخدام أصناف مقاومة تحمل الرمز VFN.",
               "تعقيم الشمسي للتربة 6 أسابيع صيفا (غطاء بلاستيكي على التربة الرطبة).",
               "دوران محصولي 4 سنوات على الأقل. تجنب إعادة الزراعة في الأراضي الملوثة."),
    },
    ("tomato", "blossom_end_rot"): {
        "en": ("calcium nitrate (foliar)",
               "Foliar spray of calcium nitrate at 5 g/L at fruit set stage. Repeat every 10 days.",
               "Maintain consistent irrigation to improve calcium uptake. Avoid water stress.",
               "Mulch to regulate soil moisture. This is a physiological disorder, not a pathogen."),
        "fr": ("nitrate de calcium (foliaire)",
               "Pulvérisation foliaire de nitrate de calcium à 5 g/L à la nouaison. Répéter toutes les 10 jours.",
               "Maintenir une irrigation régulière pour favoriser l'absorption du calcium.",
               "Pailler pour réguler l'humidité du sol. Il s'agit d'un trouble physiologique, pas d'un pathogène."),
        "ar": ("نترات الكالسيوم (رش ورقي)",
               "رش ورقي بنترات الكالسيوم بـ5 غ/ل عند مرحلة العقد. التكرار كل 10 أيام.",
               "الحفاظ على ري منتظم لتحسين امتصاص الكالسيوم. تجنب الإجهاد المائي.",
               "التغطية بالمهاد لتنظيم رطوبة التربة. هذا اضطراب فيزيولوجي وليس مرضا معدِيا."),
    },
    # ==== WHEAT ====
    ("wheat", "septoria_tritici"): {
        "en": ("tebuconazole 250 EC",
               "Apply tebuconazole 250 EC at 1 L/ha at flag leaf stage (GS37–39).",
               "Second spray at heading (GS55) if humid conditions persist.",
               "Grow resistant varieties (Boussalem). Bury crop residues after harvest."),
        "fr": ("tébuconazole 250 EC",
               "Tébuconazole 250 EC à 1 L/ha au stade feuille étendard (GS37–39).",
               "Deuxième application à l'épiaison (GS55) si temps humide persistant.",
               "Variétés résistantes (Boussalem). Enfouir les résidus après récolte."),
        "ar": ("تيبوكونازول 250 EC",
               "رش تيبوكونازول 250 EC بـ1 ل/هكتار عند مرحلة الورقة العلم (GS37-39).",
               "رشة ثانية عند الإسبال (GS55) في حال استمرار الطقس الرطب.",
               "أصناف مقاومة (بوسالم). دفن مخلفات المحصول بعد الحصاد."),
    },
    ("wheat", "yellow_rust"): {
        "en": ("tebuconazole 250 EC, propiconazole",
               "Apply tebuconazole + propiconazole at first pustule appearance.",
               "Second spray 14 days later if infection continues to spread.",
               "Monitor fields weekly from February. Use resistant varieties when available."),
        "fr": ("tébuconazole 250 EC, propiconazole",
               "Tébuconazole + propiconazole dès l'apparition des premières pustules.",
               "Deuxième application 14 jours après si la maladie progresse.",
               "Surveiller les parcelles chaque semaine dès février. Choisir des variétés résistantes."),
        "ar": ("تيبوكونازول 250 EC، بروبيكونازول",
               "رش تيبوكونازول + بروبيكونازول عند ظهور أولى البثور.",
               "رشة ثانية بعد 14 يوما إذا استمر تطور المرض.",
               "مراقبة الحقول أسبوعيا من فبراير. استخدام أصناف مقاومة عند توفرها."),
    },
    ("wheat", "brown_rust"): {
        "en": ("tebuconazole 250 EC",
               "Tebuconazole 250 EC at 1 L/ha at first symptoms on upper leaves.",
               "Second spray 14 days later if disease pressure remains high.",
               "Same resistant varieties as for yellow rust. Crop monitoring from late March."),
        "fr": ("tébuconazole 250 EC",
               "Tébuconazole 250 EC à 1 L/ha dès les premiers symptômes sur feuilles supérieures.",
               "Deuxième application 14 jours après si la pression reste forte.",
               "Mêmes variétés résistantes que pour la rouille jaune. Surveillance à partir de fin mars."),
        "ar": ("تيبوكونازول 250 EC",
               "تيبوكونازول 250 EC بـ1 ل/هكتار عند ظهور الأعراض الأولى على الأوراق العليا.",
               "رشة ثانية بعد 14 يوما إذا استمر الضغط.",
               "نفس الأصناف المقاومة كالصدأ الأصفر. مراقبة الحقول من أواخر مارس."),
    },
    ("wheat", "fusarium_head_blight"): {
        "en": ("prothioconazole",
               "Prothioconazole at flowering (50% anthesis) — critical timing window.",
               "A second application 5–7 days later improves control under high humidity.",
               "Avoid wheat-after-corn rotation. Bury residues. Choose tolerant varieties."),
        "fr": ("prothioconazole",
               "Prothioconazole à la floraison (50% d'anthèse) — fenêtre d'application critique.",
               "Une deuxième application 5–7 jours après améliore l'efficacité par temps humide.",
               "Éviter la rotation blé-maïs. Enfouir les résidus. Choisir des variétés tolérantes."),
        "ar": ("بروثيوكونازول",
               "رش بروثيوكونازول عند الإزهار (50% إزهار) — النافذة الزمنية الحرجة.",
               "رشة ثانية بعد 5-7 أيام تحسّن الفعالية في الطقس الرطب.",
               "تجنب تعاقب القمح بعد الذرة. دفن المخلفات. اختيار أصناف متحملة."),
    },
    ("wheat", "hessian_fly"): {
        "en": ("phosmet (at tillering if outbreak)",
               "Delay autumn sowing until after the first autumn rains ('Hessian fly-free date').",
               "Apply phosmet at tillering only if adult fly population is confirmed above threshold.",
               "Use resistant varieties. Destroy volunteer wheat plants around fields."),
        "fr": ("phosmet (au tallage si attaque confirmée)",
               "Retarder les semis d'automne après les premières pluies (date sans mouche de Hesse).",
               "Phosmet au tallage uniquement si la population adulte dépasse le seuil de nuisibilité.",
               "Utiliser des variétés résistantes. Détruire les repousses de blé autour des parcelles."),
        "ar": ("فوسميت (عند الإشطاء إذا تأكدت الإصابة)",
               "تأخير بذر الخريف إلى ما بعد أمطار الخريف الأولى (تاريخ الأمان من الذبابة).",
               "رش فوسميت عند الإشطاء فقط إذا تجاوزت أعداد الحشرة البالغة عتبة الضرر.",
               "زراعة أصناف مقاومة. إتلاف نباتات القمح الطوعية حول الحقول."),
    },
    # ==== OLIVE ====
    ("olive", "peacock_spot"): {
        "en": ("copper oxychloride 50% WP",
               "Copper oxychloride 50% WP at 4 kg/ha before winter rains and after flowering. Repeat every 21 days.",
               "Spray must cover undersides of leaves where spores overwinter.",
               "Prune to improve air circulation. Collect and burn fallen infected leaves."),
        "fr": ("oxychlorure de cuivre 50% WP",
               "Oxychlorure de cuivre 50% WP à 4 kg/ha avant les pluies hivernales et après la floraison. Répéter toutes les 21 jours.",
               "Bien couvrir la face inférieure des feuilles où les spores hivernent.",
               "Tailler pour améliorer la ventilation. Ramasser et brûler les feuilles tombées infectées."),
        "ar": ("أوكسيكلوريد النحاس 50% WP",
               "أوكسيكلوريد النحاس 50% WP بـ4 كغ/هكتار قبل أمطار الشتاء وبعد الإزهار. التكرار كل 21 يوما.",
               "يجب تغطية الوجه السفلي للأوراق حيث تشتي الجراثيم.",
               "التقليم لتحسين التهوية. جمع الأوراق المصابة المتساقطة وحرقها."),
    },
    ("olive", "olive_fly"): {
        "en": ("spinosad (bait spray), kaolin clay",
               "Spinosad bait sprays every 14 days from July. Apply kaolin clay film as physical preventive.",
               "McPhail traps baited with ammonium carbonate for monitoring and mass trapping.",
               "Harvest early to reduce fruit infestation. Remove fallen fruit from ground."),
        "fr": ("spinosad (appât), kaolin",
               "Appâts à base de spinosad toutes les 2 semaines à partir de juillet. Film de kaolin en préventif.",
               "Pièges McPhail avec carbonate d'ammonium pour suivi et piégeage massif.",
               "Récolter précocement pour réduire l'infestation. Ramasser les olives tombées."),
        "ar": ("سبينوساد (طُعم)، كاولين",
               "رش طُعم سبينوساد كل 14 يوما ابتداء من يوليو. طبقة كاولين وقائية على الثمار.",
               "مصائد ماكفيل ببيكربونات الأمونيوم للرصد والمكافحة الجماعية.",
               "الحصاد المبكر للحد من الإصابة. جمع الثمار الساقطة من التربة."),
    },
    ("olive", "olive_knot"): {
        "en": ("copper hydroxide 50% WP",
               "Copper hydroxide 50% WP at 3 kg/ha immediately after pruning and after hail events.",
               "Prune and burn galled branches during dry weather to prevent spore spread.",
               "Disinfect pruning tools with 10% bleach between trees."),
        "fr": ("hydroxyde de cuivre 50% WP",
               "Hydroxyde de cuivre 50% WP à 3 kg/ha immédiatement après la taille et après grêle.",
               "Tailler et brûler les rameaux tuberculés par temps sec pour limiter la dispersion des spores.",
               "Désinfecter les outils de taille à l'eau de Javel 10% entre chaque arbre."),
        "ar": ("هيدروكسيد النحاس 50% WP",
               "هيدروكسيد النحاس 50% WP بـ3 كغ/هكتار فور انتهاء التقليم وبعد حوادث البرد.",
               "تقليم وحرق الأفرع المصابة بالسل في الطقس الجاف لمنع انتشار الجراثيم.",
               "تعقيم أدوات التقليم بمحلول بليتش 10% بين كل شجرة وأخرى."),
    },
    ("olive", "verticillium_wilt"): {
        "en": ("no chemical cure",
               "No effective chemical treatment. Plant tolerant varieties: Chemlal, Sigoise.",
               "Soil solarization before planting in infested fields.",
               "Avoid intercropping with tomato, cotton, or potato which share the same pathogen."),
        "fr": ("aucun traitement chimique efficace",
               "Aucun traitement chimique efficace. Planter des variétés tolérantes : Chemlal, Sigoise.",
               "Solarisation du sol avant plantation sur parcelles contaminées.",
               "Éviter l'association culturale avec tomate, coton ou pomme de terre qui partagent le pathogène."),
        "ar": ("لا علاج كيميائي فعّال",
               "لا علاج كيميائي فعّال. زراعة أصناف متحملة: شملال، سيقواز.",
               "التعقيم الشمسي للتربة قبل الزراعة في الأراضي الملوثة.",
               "تجنب زراعة الطماطم والقطن والبطاطا في نفس القطعة لاشتراكها في العائل."),
    },
    ("olive", "anthracnose"): {
        "en": ("copper oxychloride 50% WP",
               "Copper oxychloride sprays from October, before and during harvest.",
               "Harvest early — delay increases fruit infection rate significantly.",
               "Remove fallen fruit from ground. Improve drainage under canopy."),
        "fr": ("oxychlorure de cuivre 50% WP",
               "Traitements à l'oxychlorure de cuivre à partir d'octobre, avant et pendant la récolte.",
               "Récolter tôt — le retard augmente sensiblement le taux d'infection des fruits.",
               "Ramasser les olives tombées. Améliorer le drainage sous la frondaison."),
        "ar": ("أوكسيكلوريد النحاس 50% WP",
               "رش أوكسيكلوريد النحاس ابتداء من أكتوبر قبل الحصاد وخلاله.",
               "الحصاد المبكر — التأخير يزيد نسبة إصابة الثمار بشكل ملحوظ.",
               "جمع الثمار الساقطة. تحسين الصرف تحت مظلة الأشجار."),
    },
    # ==== DATE PALM ====
    ("date_palm", "bayoud"): {
        "en": ("no chemical cure — containment only",
               "NO CURE. Immediately destroy and burn all infected trees. Do not compost.",
               "Establish strict quarantine zone. Disinfect tools and footwear on entry/exit.",
               "Plant only certified resistant varieties: Takerboucht, Bent Qbala. Contact ITDAS."),
        "fr": ("aucun traitement — confinement uniquement",
               "AUCUN TRAITEMENT. Détruire et brûler immédiatement tous les arbres infectés. Ne pas composter.",
               "Mettre en place une zone de quarantaine stricte. Désinfecter les outils et chaussures.",
               "Planter uniquement des variétés résistantes certifiées : Takerboucht, Bent Qbala. Contacter l'ITDAS."),
        "ar": ("لا علاج — احتواء فقط",
               "لا علاج. تدمير وحرق جميع النخيل المصاب فورا. عدم التسميد بمخلفاتها.",
               "إنشاء منطقة حجر صحي صارمة. تعقيم الأدوات والأحذية عند الدخول والخروج.",
               "زراعة أصناف مقاومة معتمدة فقط: تكربوشت، بنت قبالة. التواصل مع ITDAS."),
    },
    ("date_palm", "red_palm_weevil"): {
        "en": ("imidacloprid 200 SL (trunk injection), ferrugineol pheromone (traps)",
               "Install ferrugineol pheromone traps (1 per 5 ha) for early detection.",
               "Trunk injection of imidacloprid 200 SL at 10 ml per tree at infestation signs.",
               "Remove and destroy heavily infested trees. Inspect crown monthly from April."),
        "fr": ("imidaclopride 200 SL (injection tronc), phérormone ferrugineol (pièges)",
               "Installer des pièges à phéromone ferrugineol (1 par 5 ha) pour la détection précoce.",
               "Injection de tronc à l'imidaclopride 200 SL à 10 ml par arbre aux premiers signes.",
               "Abattre et détruire les arbres fortement infestés. Inspecter la couronne chaque mois d'avril."),
        "ar": ("إيميداكلوبريد 200 SL (حقن الجذع)، فيرومون الفيروجينيول (مصائد)",
               "نصب مصائد فيرومون الفيروجينيول (1 لكل 5 هكتار) للكشف المبكر.",
               "حقن الجذع بإيميداكلوبريد 200 SL بـ10 مل/شجرة عند ظهور الأعراض.",
               "إزالة وإتلاف النخيل الشديد الإصابة. فحص القمة الشهري ابتداء من أبريل."),
    },
    ("date_palm", "brown_spot"): {
        "en": ("copper oxychloride 50% WP",
               "Copper oxychloride 50% WP at 0.4% concentration sprayed on inflorescences.",
               "Apply at pollination and repeat at fruit set if disease pressure is high.",
               "Remove and burn infected fronds. Avoid wetting fronds during irrigation."),
        "fr": ("oxychlorure de cuivre 50% WP",
               "Oxychlorure de cuivre 50% WP à 0.4% de concentration sur les inflorescences.",
               "Appliquer à la pollinisation et répéter à la nouaison si pression forte.",
               "Enlever et brûler les palmes infectées. Éviter de mouiller les palmes à l'irrigation."),
        "ar": ("أوكسيكلوريد النحاس 50% WP",
               "رش أوكسيكلوريد النحاس 50% WP بتركيز 0.4% على النورات الزهرية.",
               "التطبيق عند التلقيح والتكرار عند العقد إذا كان الضغط شديدا.",
               "إزالة وحرق الجريد المصاب. تجنب ترطيب الجريد أثناء الري."),
    },
    ("date_palm", "khamedj"): {
        "en": ("copper oxychloride 50% WP",
               "Copper oxychloride sprays at pollination time (February–March).",
               "Prune and burn all infected inflorescences in spring before spore release.",
               "Use healthy pollen from disease-free palms. Bag female inflorescences early."),
        "fr": ("oxychlorure de cuivre 50% WP",
               "Traitements au cuivre au moment de la pollinisation (février–mars).",
               "Tailler et brûler toutes les inflorescences infectées au printemps avant libération des spores.",
               "Utiliser du pollen sain provenant de palmiers indemnes. Ensacher tôt les inflorescences femelles."),
        "ar": ("أوكسيكلوريد النحاس 50% WP",
               "رش النحاس عند التلقيح (فبراير-مارس).",
               "قطع وحرق جميع النورات المصابة في الربيع قبل انطلاق الجراثيم.",
               "استخدام حبوب لقاح سليمة من نخيل خالٍ من المرض. تكييس النورات الأنثوية مبكرا."),
    },
    ("date_palm", "boufaroua_mite"): {
        "en": ("wettable sulfur 80% WP, abamectin 1.8 EC",
               "Wettable sulfur 80% WP at 5 g/L sprayed on bunches in June and July.",
               "Abamectin 1.8 EC at 0.4 L/ha as alternative if sulfur tolerance is poor.",
               "Avoid dusty conditions which promote mite buildup. Bag bunches with fine netting."),
        "fr": ("soufre mouillable 80% WP, abamectine 1.8 EC",
               "Soufre mouillable 80% WP à 5 g/L sur les régimes en juin et juillet.",
               "Abamectine 1.8 EC à 0.4 L/ha en alternative si intolérance au soufre.",
               "Éviter les conditions poussiéreuses qui favorisent les acariens. Ensacher les régimes."),
        "ar": ("كبريت مبلل 80% WP، أباميكتين 1.8 EC",
               "رش كبريت مبلل 80% WP بـ5 غ/ل على العراجين في يونيو ويوليو.",
               "أباميكتين 1.8 EC بـ0.4 ل/هكتار كبديل عند ضعف تحمل الكبريت.",
               "تجنب الظروف المغبرة التي تشجع على تكاثر الأكاروس. تكييس العراجين بشبكة دقيقة."),
    },
    # ==== POTATO ====
    ("potato", "late_blight"): {
        "en": ("mancozeb 80% WP, metalaxyl + mancozeb",
               "Mancozeb 80% WP at 2 kg/ha preventively every 7 days during cool humid weather.",
               "At first symptoms: switch to metalaxyl + mancozeb (systemic) combination.",
               "Use certified blight-free seed tubers. Hilling improves tuber protection."),
        "fr": ("mancozèbe 80% WP, métalaxyl + mancozèbe",
               "Mancozèbe 80% WP à 2 kg/ha en préventif toutes les 7 jours par temps froid et humide.",
               "Aux premiers symptômes : passer au mélange métalaxyl + mancozèbe (systémique).",
               "Utiliser des tubercules-semences certifiés sans mildiou. Le buttage protège les tubercules."),
        "ar": ("مانكوزيب 80% WP، ميتالاكسيل + مانكوزيب",
               "مانكوزيب 80% WP بـ2 كغ/هكتار وقائيا كل 7 أيام في الطقس البارد الرطب.",
               "عند ظهور الأعراض: التحول إلى مزيج ميتالاكسيل + مانكوزيب (جهازي).",
               "استخدام درنات بذر معتمدة خالية من المرض. التتريب يحمي الدرنات."),
    },
    ("potato", "early_blight"): {
        "en": ("chlorothalonil 75% WP",
               "Chlorothalonil 75% WP at 2 kg/ha every 10 days starting from tuber bulking.",
               "Increase frequency to 7 days under hot humid pressure.",
               "Avoid overhead irrigation. Ensure adequate potassium nutrition."),
        "fr": ("chlorothalonil 75% WP",
               "Chlorothalonil 75% WP à 2 kg/ha toutes les 10 jours dès la tubérisation.",
               "Réduire l'intervalle à 7 jours sous forte pression chaude et humide.",
               "Éviter l'irrigation par aspersion. Assurer un apport suffisant en potassium."),
        "ar": ("كلوروثالونيل 75% WP",
               "كلوروثالونيل 75% WP بـ2 كغ/هكتار كل 10 أيام ابتداء من مرحلة تضخم الدرنات.",
               "تقصير الفترة إلى 7 أيام في الطقس الحار الرطب.",
               "تجنب الري بالرش الرأسي. ضمان تغذية كافية بالبوتاسيوم."),
    },
    ("potato", "potato_virus_y"): {
        "en": ("imidacloprid (to control aphid vectors)",
               "No chemical treatment for the virus. Use only certified disease-free seed tubers.",
               "Control aphid vectors with imidacloprid 200 SL at 0.5 L/ha.",
               "Remove infected plants immediately. Reflective mulch repels aphids."),
        "fr": ("imidaclopride (contre les pucerons vecteurs)",
               "Aucun traitement chimique contre le virus. Utiliser uniquement des semences certifiées indemnes.",
               "Contrôler les pucerons vecteurs avec imidaclopride 200 SL à 0.5 L/ha.",
               "Arracher les plants infectés immédiatement. Paillage réfléchissant pour repousser les pucerons."),
        "ar": ("إيميداكلوبريد (لمكافحة حشرة المن الناقلة)",
               "لا علاج كيميائي للفيروس. استخدام درنات بذر معتمدة خالية من المرض فقط.",
               "مكافحة حشرة المن الناقلة بإيميداكلوبريد 200 SL بـ0.5 ل/هكتار.",
               "اقتلاع النباتات المصابة فورا. المهاد العاكس يطرد حشرة المن."),
    },
    ("potato", "colorado_beetle"): {
        "en": ("spinosad 480 SC",
               "Spinosad 480 SC at 0.1 L/ha at first larval appearance. Repeat after 7 days if needed.",
               "Hand-pick egg masses and larvae on small plots (under 1 ha).",
               "Crop rotation minimum 3 years. Deep ploughing in autumn exposes overwintering adults."),
        "fr": ("spinosad 480 SC",
               "Spinosad 480 SC à 0.1 L/ha à l'apparition des premières larves. Répéter après 7 jours si besoin.",
               "Ramasser manuellement les pontes et larves sur petites surfaces (< 1 ha).",
               "Rotation culturale de 3 ans minimum. Labour profond en automne expose les adultes hivernants."),
        "ar": ("سبينوساد 480 SC",
               "سبينوساد 480 SC بـ0.1 ل/هكتار عند ظهور أولى اليرقات. التكرار بعد 7 أيام إذا لزم.",
               "جمع أكتاف البيض واليرقات يدويا على المساحات الصغيرة (أقل من هكتار).",
               "دوران محصولي 3 سنوات على الأقل. الحرث العميق خريفا يكشف الحشرات البالغة الشتوية."),
    },
    ("potato", "blackleg"): {
        "en": ("no effective chemical treatment",
               "Use only certified seeds. Treat cut surfaces with copper-based powder before planting.",
               "Avoid harvesting in wet conditions which spread bacteria.",
               "Cure tubers 10 days at 15°C before storage to allow cut surfaces to suberize."),
        "fr": ("aucun traitement chimique efficace",
               "Utiliser uniquement des semences certifiées. Traiter les surfaces de coupe avec une poudre cuivrique avant plantation.",
               "Éviter la récolte par temps humide qui propage la bactérie.",
               "Ressuer les tubercules 10 jours à 15°C avant stockage pour suberiser les surfaces."),
        "ar": ("لا علاج كيميائي فعّال",
               "استخدام بذور معتمدة فقط. معالجة أسطح القطع بمسحوق نحاسي قبل الزراعة.",
               "تجنب الحصاد في الطقس الرطب الذي ينشر البكتيريا.",
               "تعتيق الدرنات 10 أيام عند 15°م قبل التخزين لتكوين طبقة فلينية على الجروح."),
    },
    # ==== ONION ====
    ("onion", "downy_mildew"): {
        "en": ("mancozeb 80% WP, metalaxyl",
               "Mancozeb 80% WP at 2.5 kg/ha + metalaxyl every 10 days from bulbing stage.",
               "Avoid overhead irrigation which spreads spores. Irrigate early morning.",
               "Improve row spacing for air circulation. Destroy infected crop debris."),
        "fr": ("mancozèbe 80% WP, métalaxyl",
               "Mancozèbe 80% WP à 2.5 kg/ha + métalaxyl toutes les 10 jours dès la bulbaison.",
               "Éviter l'irrigation par aspersion qui disperse les spores. Irriguer le matin.",
               "Augmenter l'espacement pour la ventilation. Détruire les débris végétaux infectés."),
        "ar": ("مانكوزيب 80% WP، ميتالاكسيل",
               "مانكوزيب 80% WP بـ2.5 كغ/هكتار + ميتالاكسيل كل 10 أيام ابتداء من مرحلة تكوين البصلة.",
               "تجنب الري بالرش الذي ينشر الجراثيم. الري في الصباح الباكر.",
               "توسيع المسافة بين الصفوف للتهوية. إتلاف مخلفات المحصول المصاب."),
    },
    ("onion", "purple_blotch"): {
        "en": ("chlorothalonil 75% WP",
               "Chlorothalonil 75% WP at 2 kg/ha every 7 days during warm humid weather.",
               "Combine with mancozeb rotation to prevent resistance development.",
               "Remove infected leaves. Avoid excessive nitrogen which promotes susceptibility."),
        "fr": ("chlorothalonil 75% WP",
               "Chlorothalonil 75% WP à 2 kg/ha tous les 7 jours par temps chaud et humide.",
               "Alterner avec mancozèbe pour prévenir les résistances.",
               "Enlever les feuilles infectées. Éviter l'excès d'azote qui favorise la sensibilité."),
        "ar": ("كلوروثالونيل 75% WP",
               "كلوروثالونيل 75% WP بـ2 كغ/هكتار كل 7 أيام في الطقس الحار الرطب.",
               "التناوب مع المانكوزيب لمنع تطور المقاومة.",
               "إزالة الأوراق المصابة. تجنب زيادة النيتروجين التي تزيد الحساسية."),
    },
    ("onion", "fusarium_basal_rot"): {
        "en": ("no chemical cure",
               "No effective chemical cure. Practice long crop rotation of 5 years minimum.",
               "Avoid mechanical injury at harvest — wounds are entry points for the fungus.",
               "Plant on raised beds with good drainage. Use healthy certified transplants."),
        "fr": ("aucun traitement chimique efficace",
               "Aucun traitement chimique efficace. Rotation longue de 5 ans minimum.",
               "Éviter les blessures mécaniques à la récolte — elles sont des portes d'entrée pour le champignon.",
               "Planter sur billons bien drainés. Utiliser des transplants sains certifiés."),
        "ar": ("لا علاج كيميائي فعّال",
               "لا علاج كيميائي فعّال. دوران محصولي 5 سنوات على الأقل.",
               "تجنب الجروح الميكانيكية عند الحصاد — فهي مداخل للفطر.",
               "الزراعة على أحواض مرتفعة جيدة الصرف. استخدام شتلات معتمدة سليمة."),
    },
    ("onion", "thrips"): {
        "en": ("spinosad 480 SC",
               "Spinosad 480 SC at 0.15 L/ha. Repeat every 7 days under heavy pressure.",
               "Apply in the morning when thrips are active on leaf surfaces.",
               "Reflective silver mulch repels thrips. Remove weed hosts around the field."),
        "fr": ("spinosad 480 SC",
               "Spinosad 480 SC à 0.15 L/ha. Répéter toutes les 7 jours sous forte pression.",
               "Appliquer le matin quand les thrips sont actifs sur les feuilles.",
               "Paillage réfléchissant argenté pour repousser les thrips. Désherber les abords."),
        "ar": ("سبينوساد 480 SC",
               "سبينوساد 480 SC بـ0.15 ل/هكتار. التكرار كل 7 أيام في حالة الضغط الشديد.",
               "الرش صباحا عندما تكون التربس نشطة على الأوراق.",
               "المهاد الفضي العاكس يطرد التربس. إزالة الأعشاب الضارة المحيطة بالحقل."),
    },
    ("onion", "white_rot"): {
        "en": ("tebuconazole (soil drench)",
               "Long rotation of 8+ years is the only reliable control measure.",
               "Tebuconazole soil drench at planting only in fields with confirmed history of the disease.",
               "Avoid moving soil from infested fields. The pathogen survives in soil for 20+ years."),
        "fr": ("tébuconazole (drench sol)",
               "Rotation longue de 8+ ans — seule mesure réellement efficace.",
               "Drench au tébuconazole à la plantation uniquement sur parcelles à historique confirmé.",
               "Ne pas déplacer la terre des parcelles infestées. Le pathogène survit 20+ ans dans le sol."),
        "ar": ("تيبوكونازول (رش التربة)",
               "دوران محصولي 8 سنوات أو أكثر — الإجراء الوحيد الفعّال فعليا.",
               "رش التربة بتيبوكونازول عند الزراعة فقط في الحقول ذات التاريخ المعروف.",
               "عدم نقل التربة من الحقول الملوثة. يبقى الممرض حيّا في التربة أكثر من 20 سنة."),
    },
    # ==== PEPPER ====
    ("pepper", "phytophthora_capsici"): {
        "en": ("metalaxyl + mancozeb",
               "Metalaxyl + mancozeb drench at planting. Repeat every 21 days if wet conditions persist.",
               "Plant on raised beds to improve drainage. Avoid waterlogging at all growth stages.",
               "Remove infected plants immediately. Do not replant susceptible crops for 3 years."),
        "fr": ("métalaxyl + mancozèbe",
               "Drench métalaxyl + mancozèbe à la plantation. Répéter toutes les 21 jours si temps humide.",
               "Planter sur billons pour améliorer le drainage. Éviter l'engorgement à tous les stades.",
               "Arracher les plantes infectées immédiatement. Ne pas replanter de cultures sensibles pendant 3 ans."),
        "ar": ("ميتالاكسيل + مانكوزيب",
               "رش التربة بميتالاكسيل + مانكوزيب عند الزراعة. التكرار كل 21 يوما في الطقس الرطب.",
               "الزراعة على أحواض مرتفعة لتحسين الصرف. تجنب تراكم الماء في جميع مراحل النمو.",
               "اقتلاع النباتات المصابة فورا. عدم زراعة محاصيل حساسة لمدة 3 سنوات."),
    },
    ("pepper", "anthracnose"): {
        "en": ("copper oxychloride 50% WP, chlorothalonil 75% WP",
               "Copper oxychloride 50% WP at 3 kg/ha. Alternate with chlorothalonil every 7 days near harvest.",
               "Spray must cover fruit surface — lesions appear on ripe and ripening fruit.",
               "Harvest at correct maturity. Avoid wounding fruit during harvest operations."),
        "fr": ("oxychlorure de cuivre 50% WP, chlorothalonil 75% WP",
               "Oxychlorure de cuivre 50% WP à 3 kg/ha, en alternance avec chlorothalonil tous les 7 jours à l'approche de la récolte.",
               "Bien couvrir la surface des fruits — les lésions apparaissent sur fruits mûrs.",
               "Récolter à la bonne maturité. Éviter de blesser les fruits lors de la récolte."),
        "ar": ("أوكسيكلوريد النحاس 50% WP، كلوروثالونيل 75% WP",
               "أوكسيكلوريد النحاس 50% WP بـ3 كغ/هكتار، يتناوب مع كلوروثالونيل كل 7 أيام قرب الحصاد.",
               "يجب تغطية سطح الثمار جيدا — تظهر الآفة على الثمار الناضجة.",
               "الحصاد عند النضج الصحيح. تجنب جرح الثمار أثناء عمليات الحصاد."),
    },
    ("pepper", "tswv"): {
        "en": ("spinosad (for thrip vectors)",
               "No chemical treatment for the virus. Control thrip vectors with spinosad 480 SC at 0.15 L/ha.",
               "Remove and destroy infected plants. Use insect-proof netting in nursery.",
               "Plant resistant varieties where available. Avoid planting near other solanaceous crops."),
        "fr": ("spinosad (contre thrips vecteurs)",
               "Aucun traitement contre le virus. Contrôler les thrips vecteurs avec spinosad 480 SC à 0.15 L/ha.",
               "Arracher et détruire les plants malades. Filets anti-insectes en pépinière.",
               "Planter des variétés résistantes si disponibles. Éviter la proximité des autres solanacées."),
        "ar": ("سبينوساد (لمكافحة التربس الناقلة)",
               "لا علاج كيميائي للفيروس. مكافحة التربس الناقلة بسبينوساد 480 SC بـ0.15 ل/هكتار.",
               "اقتلاع وإتلاف النباتات المصابة. شبكات مضادة للحشرات في المشتل.",
               "زراعة أصناف مقاومة إذا توفرت. تجنب الزراعة بالقرب من محاصيل الباذنجانيات."),
    },
    ("pepper", "bacterial_spot"): {
        "en": ("copper hydroxide 50% WP, mancozeb 80% WP",
               "Copper hydroxide 50% WP at 3 kg/ha + mancozeb 80% WP every 7 days in warm wet weather.",
               "Start sprays preventively at transplant. Avoid overhead irrigation.",
               "Use disease-free certified seeds. Disinfect transplant trays between crops."),
        "fr": ("hydroxyde de cuivre 50% WP, mancozèbe 80% WP",
               "Hydroxyde de cuivre 50% WP à 3 kg/ha + mancozèbe 80% WP toutes les 7 jours par temps chaud et humide.",
               "Démarrer en préventif dès la transplantation. Éviter l'irrigation par aspersion.",
               "Utiliser des graines certifiées indemnes. Désinfecter les plateaux entre les cultures."),
        "ar": ("هيدروكسيد النحاس 50% WP، مانكوزيب 80% WP",
               "هيدروكسيد النحاس 50% WP بـ3 كغ/هكتار + مانكوزيب 80% WP كل 7 أيام في الطقس الحار الرطب.",
               "البدء بالرش الوقائي من الشتل. تجنب الري بالرش الرأسي.",
               "استخدام بذور معتمدة خالية من المرض. تعقيم الأصاصات بين المحاصيل."),
    },
    ("pepper", "aphids"): {
        "en": ("imidacloprid 200 SL",
               "Imidacloprid 200 SL at 0.4 L/ha foliar spray at first colony detection.",
               "Repeat after 10 days if population persists. Rotate insecticide class to prevent resistance.",
               "Reflective silver mulch deters aphids. Release lacewing (Chrysoperla) as biocontrol."),
        "fr": ("imidaclopride 200 SL",
               "Imidaclopride 200 SL à 0.4 L/ha en pulvérisation foliaire à la détection des premières colonies.",
               "Répéter après 10 jours si la population persiste. Alterner les familles d'insecticides.",
               "Paillage réfléchissant argenté. Lâcher de chrysope (Chrysoperla) en lutte biologique."),
        "ar": ("إيميداكلوبريد 200 SL",
               "رش إيميداكلوبريد 200 SL بـ0.4 ل/هكتار ورقيا عند اكتشاف أولى المستعمرات.",
               "التكرار بعد 10 أيام إذا استمر التعداد. التناوب بين مجموعات المبيدات لمنع المقاومة.",
               "المهاد الفضي العاكس يصرف المن. إطلاق أسد المن (كريزوبيرلا) للمكافحة البيولوجية."),
    },
    # ==== WATERMELON ====
    ("watermelon", "fusarium_wilt"): {
        "en": ("no chemical cure",
               "No effective chemical cure. Use certified resistant varieties as first line of defense.",
               "Soil solarization for 6 weeks during summer before planting.",
               "Long crop rotation of 5+ years with non-cucurbit crops."),
        "fr": ("aucun traitement chimique efficace",
               "Aucun traitement chimique efficace. Utiliser des variétés résistantes certifiées en priorité.",
               "Solarisation du sol 6 semaines en été avant la plantation.",
               "Rotation longue de 5+ ans avec des cultures hors cucurbitacées."),
        "ar": ("لا علاج كيميائي فعّال",
               "لا علاج كيميائي فعّال. زراعة أصناف مقاومة معتمدة كأول إجراء دفاعي.",
               "التعقيم الشمسي للتربة 6 أسابيع صيفا قبل الزراعة.",
               "دوران محصولي 5 سنوات أو أكثر مع محاصيل غير قرعية."),
    },
    ("watermelon", "anthracnose"): {
        "en": ("copper oxychloride 50% WP, chlorothalonil 75% WP",
               "Copper oxychloride 50% WP at 3 kg/ha, alternating with chlorothalonil 75% WP every 10 days.",
               "Begin preventive sprays at vine spread. Cover all leaf and fruit surfaces.",
               "Crop rotation 3 years. Avoid working in field when foliage is wet."),
        "fr": ("oxychlorure de cuivre 50% WP, chlorothalonil 75% WP",
               "Oxychlorure de cuivre 50% WP à 3 kg/ha, en alternance avec chlorothalonil 75% WP toutes les 10 jours.",
               "Commencer en préventif dès l'étalement des tiges. Couvrir feuilles et fruits.",
               "Rotation de 3 ans. Éviter de travailler en parcelle quand le feuillage est humide."),
        "ar": ("أوكسيكلوريد النحاس 50% WP، كلوروثالونيل 75% WP",
               "أوكسيكلوريد النحاس 50% WP بـ3 كغ/هكتار، يتناوب مع كلوروثالونيل 75% WP كل 10 أيام.",
               "البدء بالرش الوقائي عند امتداد السيقان. تغطية جميع الأوراق والثمار.",
               "دوران محصولي 3 سنوات. تجنب العمل في الحقل عند ابتلال الأوراق."),
    },
    ("watermelon", "powdery_mildew"): {
        "en": ("wettable sulfur 80% WP, potassium bicarbonate",
               "Wettable sulfur 80% WP at 3 kg/ha at first symptoms. Repeat every 10 days.",
               "Potassium bicarbonate at 3 g/L as organic alternative or for sulfur-sensitive varieties.",
               "Avoid excessive nitrogen. Increase plant spacing to improve air circulation."),
        "fr": ("soufre mouillable 80% WP, bicarbonate de potassium",
               "Soufre mouillable 80% WP à 3 kg/ha dès les premiers symptômes. Répéter toutes les 10 jours.",
               "Bicarbonate de potassium à 3 g/L comme alternative bio ou pour variétés sensibles au soufre.",
               "Éviter l'excès d'azote. Augmenter l'espacement pour la ventilation."),
        "ar": ("كبريت مبلل 80% WP، بيكربونات البوتاسيوم",
               "كبريت مبلل 80% WP بـ3 كغ/هكتار عند ظهور الأعراض. التكرار كل 10 أيام.",
               "بيكربونات البوتاسيوم 3 غ/ل كبديل عضوي أو للأصناف الحساسة للكبريت.",
               "تجنب الزيادة في النيتروجين. توسيع المسافة بين النباتات لتحسين التهوية."),
    },
    ("watermelon", "gummy_stem_blight"): {
        "en": ("mancozeb 80% WP",
               "Mancozeb 80% WP at 2 kg/ha every 10 days from early vegetative stage.",
               "Spray base of stems and crown — primary infection sites.",
               "Crop rotation 3 years. Avoid wetting stems with overhead irrigation."),
        "fr": ("mancozèbe 80% WP",
               "Mancozèbe 80% WP à 2 kg/ha toutes les 10 jours dès le début de la croissance végétative.",
               "Traiter la base des tiges et les couronnes — sites d'infection primaires.",
               "Rotation de 3 ans. Éviter de mouiller les tiges par aspersion."),
        "ar": ("مانكوزيب 80% WP",
               "مانكوزيب 80% WP بـ2 كغ/هكتار كل 10 أيام من بداية النمو الخضري.",
               "رش قواعد الساق والتاج — مواقع الإصابة الأولى.",
               "دوران محصولي 3 سنوات. تجنب ترطيب السيقان بالري بالرش."),
    },
    ("watermelon", "aphids"): {
        "en": ("imidacloprid 200 SL",
               "Imidacloprid 200 SL at 0.4 L/ha at first colony detection.",
               "Monitor for virus symptoms as aphids are primary WMMV/WMV vectors.",
               "Reflective silver mulch deters aphids. Remove weeds from field margins."),
        "fr": ("imidaclopride 200 SL",
               "Imidaclopride 200 SL à 0.4 L/ha à la détection des premières colonies.",
               "Surveiller les symptômes viraux car les pucerons sont vecteurs de WMMV/WMV.",
               "Paillage réfléchissant argenté. Désherber les abords des parcelles."),
        "ar": ("إيميداكلوبريد 200 SL",
               "إيميداكلوبريد 200 SL بـ0.4 ل/هكتار عند اكتشاف أولى المستعمرات.",
               "مراقبة أعراض الفيروسات إذ يعد المن ناقلا رئيسيا لفيروس WMMV/WMV.",
               "المهاد الفضي العاكس يصرف المن. إزالة الأعشاب الضارة من أطراف الحقل."),
    },
    # ==== CITRUS ====
    ("citrus", "tristeza_virus"): {
        "en": ("no chemical cure",
               "NO CURE. Plant only on tristeza-tolerant rootstocks (Citrange Troyer, Swingle citrumelo).",
               "Control aphid vectors (brown citrus aphid) with imidacloprid 200 SL at 0.5 L/ha.",
               "Remove highly symptomatic trees promptly. Use certified virus-free nursery stock."),
        "fr": ("aucun traitement chimique",
               "AUCUN TRAITEMENT. Planter uniquement sur porte-greffe tolérant (Citrange Troyer, Swingle citrumelo).",
               "Contrôler les pucerons vecteurs (puceron brun) avec imidaclopride 200 SL à 0.5 L/ha.",
               "Abattre rapidement les arbres très symptomatiques. Utiliser du matériel certifié sans virus."),
        "ar": ("لا علاج كيميائي",
               "لا علاج. الزراعة على أصول متحملة فقط (سيترانج ترويار، سوينغل سيتروميلو).",
               "مكافحة المن الناقل (المن البني) بإيميداكلوبريد 200 SL بـ0.5 ل/هكتار.",
               "إزالة الأشجار الشديدة الأعراض فورا. استخدام مواد مشتل معتمدة خالية من الفيروس."),
    },
    ("citrus", "mal_secco"): {
        "en": ("copper hydroxide 50% WP",
               "Copper hydroxide sprays in autumn before rains to protect pruning wounds.",
               "Prune and burn all affected branches during dry summer weather.",
               "Disinfect pruning tools. Plant tolerant varieties (Femminello lemon is susceptible — prefer Monachello)."),
        "fr": ("hydroxyde de cuivre 50% WP",
               "Pulvérisations de cuivre en automne avant les pluies pour protéger les plaies de taille.",
               "Tailler et brûler tous les rameaux atteints par temps sec en été.",
               "Désinfecter les outils. Préférer des variétés tolérantes (Femminello est sensible — préférer Monachello)."),
        "ar": ("هيدروكسيد النحاس 50% WP",
               "رش النحاس خريفا قبل الأمطار لحماية جروح التقليم.",
               "تقليم وحرق جميع الأفرع المصابة في الطقس الجاف صيفا.",
               "تعقيم الأدوات. تفضيل أصناف متحملة (فيمينيلو حساس — يُفضّل موناكيلو)."),
    },
    ("citrus", "brown_rot"): {
        "en": ("copper oxychloride 50% WP",
               "Copper oxychloride 50% WP at 4 kg/ha in autumn before rains. Repeat after heavy rains.",
               "Improve orchard drainage to reduce standing water under canopy.",
               "Remove and destroy fallen fruit. Raise tree skirt (remove low branches) to reduce soil splash."),
        "fr": ("oxychlorure de cuivre 50% WP",
               "Oxychlorure de cuivre 50% WP à 4 kg/ha en automne avant les pluies. Répéter après pluies abondantes.",
               "Améliorer le drainage du verger pour réduire l'eau stagnante sous les arbres.",
               "Ramasser et détruire les fruits tombés. Relevage de la jupe pour réduire les éclaboussures de sol."),
        "ar": ("أوكسيكلوريد النحاس 50% WP",
               "أوكسيكلوريد النحاس 50% WP بـ4 كغ/هكتار خريفا قبل الأمطار. التكرار بعد الأمطار الغزيرة.",
               "تحسين صرف البستان للحد من تجمع الماء تحت الأشجار.",
               "جمع وإتلاف الثمار الساقطة. رفع تاج الشجرة (إزالة الأفرع السفلية) للحد من الرشاش الأرضي."),
    },
    ("citrus", "med_fruit_fly"): {
        "en": ("spinosad (bait spray), pheromone traps",
               "Spinosad bait sprays weekly from July on 1/4 of tree canopy. Rotate with protein bait.",
               "Pheromone-based mass trapping using Trimedlure dispensers (4 per ha from June).",
               "Mesh bags on small farms to protect individual fruit. Remove and bury fallen fruit."),
        "fr": ("spinosad (appât), pièges à phéromone",
               "Appâts spinosad hebdomadaires de juillet, sur 1/4 de la canopée. Alterner avec appât protéiné.",
               "Piégeage massif à phéromone Trimedlure (4 pièges/ha dès juin).",
               "Sacs en filet sur petites exploitations. Ramasser et enfouir les fruits tombés."),
        "ar": ("سبينوساد (طُعم)، مصائد فيرومونية",
               "رش طُعم سبينوساد أسبوعيا من يوليو على ربع تاج الشجرة. التناوب مع طُعم بروتيني.",
               "مصائد فيرومون تريميدلور الجماعية (4 مصائد/هكتار من يونيو).",
               "أكياس شبكية على المزارع الصغيرة لحماية الثمار. جمع ودفن الثمار الساقطة."),
    },
    ("citrus", "citrus_canker"): {
        "en": ("copper hydroxide 50% WP",
               "Copper hydroxide 50% WP at 3 kg/ha preventively every 21 days during wet season.",
               "Remove infected fruit and leaves immediately. Disinfect tools between trees.",
               "Disease-free nursery stock mandatory. Restrict movement of plant material from infected zones."),
        "fr": ("hydroxyde de cuivre 50% WP",
               "Hydroxyde de cuivre 50% WP à 3 kg/ha en préventif toutes les 21 jours en saison humide.",
               "Enlever immédiatement les fruits et feuilles infectés. Désinfecter entre les arbres.",
               "Matériel pépinière certifié obligatoire. Restreindre les mouvements de végétaux depuis zones atteintes."),
        "ar": ("هيدروكسيد النحاس 50% WP",
               "هيدروكسيد النحاس 50% WP بـ3 كغ/هكتار وقائيا كل 21 يوما في الموسم الرطب.",
               "إزالة الثمار والأوراق المصابة فورا. تعقيم الأدوات بين الأشجار.",
               "مواد المشتل المعتمدة إلزامية. تقييد حركة النباتات من المناطق الموبوءة."),
    },
    # ==== BARLEY ====
    ("barley", "net_blotch"): {
        "en": ("tebuconazole 250 EC",
               "Tebuconazole 250 EC at 1 L/ha at flag leaf stage.",
               "Second spray at heading if weather is wet and cool.",
               "Crop rotation minimum 2 years. Use resistant varieties (Saïda 183, Rihane-03)."),
        "fr": ("tébuconazole 250 EC",
               "Tébuconazole 250 EC à 1 L/ha au stade feuille étendard.",
               "Deuxième application à l'épiaison si temps froid et humide.",
               "Rotation de 2 ans minimum. Variétés résistantes (Saïda 183, Rihane-03)."),
        "ar": ("تيبوكونازول 250 EC",
               "تيبوكونازول 250 EC بـ1 ل/هكتار عند مرحلة الورقة العلم.",
               "رشة ثانية عند الإسبال في الطقس البارد الرطب.",
               "دوران محصولي سنتين على الأقل. أصناف مقاومة (سعيدة 183، ريحان-03)."),
    },
    ("barley", "barley_scald"): {
        "en": ("tebuconazole 250 EC",
               "Tebuconazole 250 EC at 1 L/ha at flag leaf stage — same timing as net blotch.",
               "Early season sprays (tillering) if disease pressure is high on young plants.",
               "Use certified treated seed. Crop rotation 2 years."),
        "fr": ("tébuconazole 250 EC",
               "Tébuconazole 250 EC à 1 L/ha au stade feuille étendard — même calendrier que l'helminthosporiose.",
               "Traitements précoces (tallage) si forte pression sur jeunes plantes.",
               "Utiliser des semences certifiées traitées. Rotation de 2 ans."),
        "ar": ("تيبوكونازول 250 EC",
               "تيبوكونازول 250 EC بـ1 ل/هكتار عند مرحلة الورقة العلم — نفس توقيت التبقع الشبكي.",
               "رشات مبكرة (الإشطاء) عند الضغط الشديد على النباتات الصغيرة.",
               "استخدام بذور معتمدة معاملة. دوران محصولي سنتين."),
    },
    ("barley", "barley_yellow_dwarf"): {
        "en": ("imidacloprid (seed treatment)",
               "No cure for the virus. Imidacloprid seed treatment is the key preventive tool.",
               "Delay autumn sowing until after the main aphid migration flight is over.",
               "Monitor aphid populations from emergence. Remove volunteer cereal plants."),
        "fr": ("imidaclopride (traitement semences)",
               "Aucun traitement contre le virus. L'enrobage imidaclopride des semences est la prévention clé.",
               "Retarder les semis d'automne après le vol migratoire principal des pucerons.",
               "Surveiller les populations de pucerons dès la levée. Détruire les repousses de céréales."),
        "ar": ("إيميداكلوبريد (معالجة البذور)",
               "لا علاج للفيروس. معالجة البذور بإيميداكلوبريد هي الإجراء الوقائي الرئيسي.",
               "تأخير البذر الخريفي إلى ما بعد انتهاء رحلة الهجرة الرئيسية لحشرة المن.",
               "مراقبة أعداد المن من الإنبات. إتلاف نباتات الحبوب الطوعية."),
    },
    ("barley", "powdery_mildew"): {
        "en": ("tebuconazole 250 EC",
               "Tebuconazole 250 EC at 1 L/ha at first symptoms on lower leaves.",
               "Early spray at tillering is more cost-effective than later applications.",
               "Use resistant varieties. Avoid excessive nitrogen fertilization."),
        "fr": ("tébuconazole 250 EC",
               "Tébuconazole 250 EC à 1 L/ha dès les premiers symptômes sur feuilles basses.",
               "Un traitement précoce au tallage est plus rentable que des applications tardives.",
               "Utiliser des variétés résistantes. Éviter l'excès d'azote."),
        "ar": ("تيبوكونازول 250 EC",
               "تيبوكونازول 250 EC بـ1 ل/هكتار عند ظهور الأعراض الأولى على الأوراق السفلية.",
               "الرش المبكر عند الإشطاء أجدى اقتصاديا من التطبيقات المتأخرة.",
               "استخدام أصناف مقاومة. تجنب الزيادة في تسميد النيتروجين."),
    },
    ("barley", "loose_smut"): {
        "en": ("carboxin + thiram (seed treatment only)",
               "Use certified seeds treated with carboxin + thiram before sowing.",
               "No in-season treatment is possible — the fungus infects at flowering inside the head.",
               "Do not save seed from infected crops. Source certified seed each season."),
        "fr": ("carboxine + thirame (traitement semences uniquement)",
               "Utiliser des semences certifiées traitées carboxine + thirame avant le semis.",
               "Aucun traitement possible en cours de saison — le champignon infecte à la floraison.",
               "Ne pas conserver des semences de cultures infectées. Acheter des semences certifiées chaque saison."),
        "ar": ("كاربوكسين + ثيرام (معالجة البذور فقط)",
               "استخدام بذور معتمدة معاملة بكاربوكسين + ثيرام قبل البذر.",
               "لا علاج ممكن خلال الموسم — الفطر يصيب عند الإزهار داخل السنبلة.",
               "عدم حفظ بذور من محاصيل مصابة. شراء بذور معتمدة كل موسم."),
    },
}

GENERIC_TREATMENT = {
    "en": "No specific treatment data available for this disease. Consult the nearest ITDAS (Institut Technique de Développement de l'Agronomie Saharienne) for a protocol adapted to your farm.",
    "fr": "Aucune donnée de traitement spécifique disponible pour cette maladie. Consultez le centre ITDAS (Institut Technique de Développement de l'Agronomie Saharienne) le plus proche pour un protocole adapté à votre exploitation.",
    "ar": "لا تتوفر بيانات علاج محددة لهذا المرض. تواصل مع أقرب مركز ITDAS (المعهد التقني لتطوير الزراعة الصحراوية) للحصول على بروتوكول مناسب لمزرعتك.",
}


def _crop_name(crop: dict, lang: str) -> str:
    return crop["names"].get(lang) or crop["names"]["en"]


def _planting(crop: dict, lang: str) -> str:
    name = _crop_name(crop, lang)
    cal = crop["planting_calendar"]
    L = LABELS[lang]
    window = (cal.get("sowing_window_north") or cal.get("transplant_window")
              or cal.get("tree_planting_window") or "—")
    cycle = cal.get("growing_cycle_days", "—")
    harvest = cal.get("harvest_window", "—")
    monthly = cal.get("monthly_calendar", {})

    if lang == "fr":
        body = (f"# {L['planting']} — {name}\n\n"
                f"En Algérie, le {name.lower()} se sème principalement entre {window}. "
                f"Le cycle dure environ {cycle} jours, avec récolte en {harvest}.\n\n"
                f"Calendrier mensuel :\n")
    elif lang == "ar":
        body = (f"# {L['planting']} — {name}\n\n"
                f"في الجزائر، تُزرع {name} عادة في الفترة {window}. "
                f"تستغرق دورة النمو {cycle} يوما، والحصاد خلال {harvest}.\n\n"
                f"التقويم الشهري:\n")
    else:
        body = (f"# {L['planting']} — {name}\n\n"
                f"In Algeria, {name.lower()} is planted mainly during {window}. "
                f"The cycle lasts about {cycle} days, with harvest in {harvest}.\n\n"
                f"Monthly calendar:\n")

    sep = "،" if lang == "ar" else ":"
    for month, stage in monthly.items():
        body += f"- {month}{sep} {stage}\n"
    return body


def _water(crop: dict, lang: str) -> str:
    name = _crop_name(crop, lang)
    w = crop["water_requirements"]
    L = LABELS[lang]
    advice = w.get(f"irrigation_advice_{lang}", w.get("irrigation_advice_fr", ""))
    total = w.get("total_mm_per_cycle") or w.get("annual_m3_per_ha") or "—"
    regime = w.get("regime", "—").replace("_", " ")
    stages = ", ".join(w.get("critical_stages", []))

    if lang == "fr":
        return (f"# {L['water']} — {name}\n\n"
                f"Besoin total : {total} mm par cycle. Régime : {regime}. "
                f"Phases critiques : {stages}.\n\nConseil pratique : {advice}")
    if lang == "ar":
        return (f"# {L['water']} — {name}\n\n"
                f"إجمالي الاحتياج: {total} مم. نظام الري: {regime}. "
                f"المراحل الحرجة: {stages}.\n\nنصيحة عملية: {advice}")
    return (f"# {L['water']} — {name}\n\n"
            f"Total: {total} mm per cycle. Regime: {regime}. "
            f"Critical stages: {stages}.")


def _diseases(crop: dict, lang: str) -> str:
    name = _crop_name(crop, lang)
    L = LABELS[lang]
    diseases = crop.get("common_diseases", [])

    if lang == "fr":
        body = f"# {L['diseases']} — {name}\n\nMaladies et ravageurs principaux du {name.lower()} en Algérie :\n\n"
        for d in diseases:
            body += f"- **{d.get('name_fr', d['id'])}** (sévérité : {d.get('severity', '—')}, saison : {d.get('season', '—')})"
            if d.get("note"):
                body += f". {d['note']}"
            body += ".\n"
        return body
    if lang == "ar":
        body = f"# {L['diseases']} — {name}\n\nأهم الأمراض والآفات على {name} في الجزائر:\n\n"
        for d in diseases:
            body += f"- **{d.get('name_ar', d['id'])}** (الخطورة: {d.get('severity', '—')}, الموسم: {d.get('season', '—')})"
            if d.get("note"):
                body += f". {d['note']}"
            body += ".\n"
        return body
    body = f"# {L['diseases']} — {name}\n\nMain diseases and pests of {name.lower()} in Algeria:\n\n"
    for d in diseases:
        body += f"- **{d['id']}** (severity: {d.get('severity','—')}, season: {d.get('season','—')}).\n"
    return body


def _yield_overview(crop: dict, lang: str) -> str:
    name = _crop_name(crop, lang)
    y = crop["average_yield"]
    npk = crop.get("fertilization_npk_kg_per_ha") or crop.get("fertilization_npk_kg_per_tree", {})
    regions = ", ".join(crop.get("main_regions_dz", []))
    varieties = ", ".join(crop.get("varieties_dz", []))
    L = LABELS[lang]
    yield_str = ", ".join(f"{k}: {v}" for k, v in y.items() if k != "unit")
    npk_str = ", ".join(f"{k}={v}" for k, v in npk.items())

    if lang == "fr":
        return (f"# {L['yield']} — {name}\n\n"
                f"Rendement : {yield_str}. Wilayas productrices : {regions}. "
                f"Variétés : {varieties}. Fertilisation NPK : {npk_str}.")
    if lang == "ar":
        return (f"# {L['yield']} — {name}\n\n"
                f"الإنتاج: {yield_str}. الولايات المنتجة: {regions}. "
                f"الأصناف: {varieties}. التسميد NPK: {npk_str}.")
    return (f"# {L['yield']} — {name}\n\n"
            f"Yield: {yield_str}. Regions: {regions}. "
            f"Varieties: {varieties}. NPK: {npk_str}.")


def _fertilization(crop: dict, lang: str) -> str:
    """Focused fertilization chunk: NPK quantities, timing, soil pH."""
    name = _crop_name(crop, lang)
    L = LABELS[lang]
    npk_ha = crop.get("fertilization_npk_kg_per_ha")
    npk_tree = crop.get("fertilization_npk_kg_per_tree")
    soil = crop.get("soil", {})
    ph = soil.get("ph_range", "—")
    is_tree = npk_tree is not None

    if npk_ha:
        n, p, k = npk_ha.get("N", 0), npk_ha.get("P", 0), npk_ha.get("K", 0)
        unit = "kg/ha"
    elif npk_tree:
        n, p, k = npk_tree.get("N", 0), npk_tree.get("P", 0), npk_tree.get("K", 0)
        unit = "kg/arbre" if lang == "fr" else ("كغ/شجرة" if lang == "ar" else "kg/tree")
    else:
        n, p, k, unit = 0, 0, 0, "—"

    if lang == "fr":
        unit_display = "kg/ha" if not is_tree else "kg/arbre"
        body = (
            f"# {L['fertilization']} — {name}\n\n"
            f"**Besoins NPK ({unit_display}) :** Azote (N) = {n}, Phosphore (P) = {p}, Potassium (K) = {k}.\n\n"
            f"**pH du sol recommandé :** {ph}.\n\n"
            f"**Calendrier d'application :**\n"
            f"- Pré-plantation : apporter la totalité du P et K + 1/3 du N en fond.\n"
            f"- Phase végétative : apporter 1/3 du N en couverture.\n"
            f"- Floraison / fructification : apporter le 1/3 restant du N.\n\n"
            f"**Conseil pratique :** Fractionner l'azote en au moins 3 apports pour réduire les pertes par "
            f"lixiviation. Sur sol acide (pH < {ph.split('-')[0] if '-' in ph else ph}), chauler avant d'apporter le phosphore."
        )
    elif lang == "ar":
        unit_display = "كغ/هكتار" if not is_tree else "كغ/شجرة"
        body = (
            f"# {L['fertilization']} — {name}\n\n"
            f"**احتياجات NPK ({unit_display}):** نيتروجين (N) = {n}، فوسفور (P) = {p}، بوتاسيوم (K) = {k}.\n\n"
            f"**درجة حموضة التربة الموصى بها:** {ph}.\n\n"
            f"**جدول التسميد:**\n"
            f"- قبل الزراعة: إضافة كامل الفوسفور والبوتاسيوم + ثلث النيتروجين كسماد أساسي.\n"
            f"- مرحلة النمو الخضري: إضافة ثلث النيتروجين تسميدا تغطيطيا.\n"
            f"- مرحلة الإزهار/الإثمار: إضافة الثلث المتبقي من النيتروجين.\n\n"
            f"**نصيحة عملية:** تجزئة النيتروجين إلى 3 دفعات على الأقل للحد من الضياع بالرشح. "
            f"في التربة الحمضية (pH أقل من {ph.split('-')[0] if '-' in ph else ph})، إضافة الجير قبل تسميد الفوسفور."
        )
    else:
        unit_display = "kg/ha" if not is_tree else "kg/tree"
        body = (
            f"# {L['fertilization']} — {name}\n\n"
            f"**NPK requirements ({unit_display}):** Nitrogen (N) = {n}, Phosphorus (P) = {p}, Potassium (K) = {k}.\n\n"
            f"**Recommended soil pH:** {ph}.\n\n"
            f"**Application schedule:**\n"
            f"- Pre-planting: apply all P and K + 1/3 of N as basal dressing.\n"
            f"- Vegetative phase: apply 1/3 of N as top dressing.\n"
            f"- Flowering / fruit set: apply remaining 1/3 of N.\n\n"
            f"**Practical advice:** Split nitrogen into at least 3 applications to reduce leaching losses. "
            f"On acidic soils (pH < {ph.split('-')[0] if '-' in ph else ph}), lime before applying phosphorus."
        )
    return body


def _treatment(crop: dict, disease: dict, lang: str) -> str:
    """One treatment chunk per disease per language."""
    crop_name = _crop_name(crop, lang)
    crop_id = crop["id"]
    disease_id = disease["id"]
    L = LABELS[lang]

    data = TREATMENTS.get((crop_id, disease_id))

    if data is None:
        if lang == "fr":
            dis_name = disease.get("name_fr", disease_id)
            return (f"# {L['treatment']} — {dis_name} sur {crop_name}\n\n"
                    f"{GENERIC_TREATMENT['fr']}")
        elif lang == "ar":
            dis_name = disease.get("name_ar", disease_id)
            return (f"# {L['treatment']} — {dis_name} على {crop_name}\n\n"
                    f"{GENERIC_TREATMENT['ar']}")
        else:
            return (f"# {L['treatment']} — {disease_id} on {crop_name}\n\n"
                    f"{GENERIC_TREATMENT['en']}")

    active, preventive, curative, cultural = data[lang]

    if lang == "fr":
        dis_name = disease.get("name_fr", disease_id)
        return (
            f"# Traitement — {dis_name} sur {crop_name}\n\n"
            f"**Culture :** {crop_name}  \n"
            f"**Matières actives :** {active}\n\n"
            f"**Prévention :** {preventive}\n\n"
            f"**Traitement curatif :** {curative}\n\n"
            f"**Lutte culturale/biologique :** {cultural}"
        )
    elif lang == "ar":
        dis_name = disease.get("name_ar", disease_id)
        return (
            f"# علاج — {dis_name} على {crop_name}\n\n"
            f"**المحصول:** {crop_name}  \n"
            f"**المواد الفعالة:** {active}\n\n"
            f"**الوقاية:** {preventive}\n\n"
            f"**العلاج الكيميائي:** {curative}\n\n"
            f"**المكافحة الزراعية/البيولوجية:** {cultural}"
        )
    else:
        return (
            f"# Treatment — {disease_id} on {crop_name}\n\n"
            f"**Crop:** {crop_name}  \n"
            f"**Active ingredients:** {active}\n\n"
            f"**Preventive:** {preventive}\n\n"
            f"**Curative:** {curative}\n\n"
            f"**Cultural/biological control:** {cultural}"
        )


# Topic -> builder function (for crop-level chunks)
TOPICS = {
    "planting": _planting,
    "water": _water,
    "diseases": _diseases,
    "yield": _yield_overview,
    "fertilization": _fertilization,
}


def build_kb(seed_path: str | None = None, out_dir: str | None = None) -> int:
    """Generate all KB chunks. Returns the number of files written."""
    seed_path = seed_path or settings.seed_json_path
    out_dir = Path(out_dir or settings.kb_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(seed_path, encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for crop_id, crop in data["crops"].items():
        # Crop-level topic chunks (planting, water, diseases, yield, fertilization)
        for topic, builder in TOPICS.items():
            for lang in ("fr", "ar", "en"):
                text = builder(crop, lang).strip() + "\n"
                (out_dir / f"{crop_id}__{topic}__{lang}.md").write_text(text, encoding="utf-8")
                count += 1

        # Per-disease treatment chunks
        for disease in crop.get("common_diseases", []):
            disease_id = disease["id"]
            for lang in ("fr", "ar", "en"):
                text = _treatment(crop, disease, lang).strip() + "\n"
                fname = f"{crop_id}_{disease_id}__treatment__{lang}.md"
                (out_dir / fname).write_text(text, encoding="utf-8")
                count += 1

    return count
