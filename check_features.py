from catboost import CatBoostClassifier

model_paths = {
    "DILI": r"C:\Users\user\Toxicology\ToxiPhar\predictor\models\DILI_catboost_production.cbm",
    "hERG": r"C:\Users\user\Toxicology\ToxiPhar\predictor\models\hERG_catboost_production.cbm",
    "CYP3A4": r"C:\Users\user\Toxicology\ToxiPhar\predictor\models\CYP3A4_catboost_production.cbm",
    "Ames": r"C:\Users\user\Toxicology\ToxiPhar\predictor\models\Ames_catboost_production.cbm",
    "Teratogenicity": r"C:\Users\user\Toxicology\ToxiPhar\predictor\models\Teratogenicity_catboost_production.cbm",
}

for name, path in model_paths.items():
    m = CatBoostClassifier()
    m.load_model(path)
    print(f"=== {name} ===")
    print("Number of features:", len(m.feature_names_))
    print("Feature names:", m.feature_names_)
    print()