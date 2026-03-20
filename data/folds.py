from sklearn.model_selection import StratifiedKFold


def make_folds(df, n_folds=5, seed=42):
    """
    Add a 'fold' column (0-4) to df using stratified split on primary_label.
    Both team members must use this exact function — never split manually.
    """
    df = df.copy()
    df['fold'] = -1

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (_, val_idx) in enumerate(
        skf.split(df, df['primary_label'].astype(str))
    ):
        df.loc[val_idx, 'fold'] = fold

    print(f"Fold distribution:")
    print(df['fold'].value_counts().sort_index())
    return df


def get_fold(df, fold):
    """Return (train_df, val_df) for a given fold number."""
    assert 'fold' in df.columns, "Run make_folds() first"
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df   = df[df['fold'] == fold].reset_index(drop=True)
    print(f"Fold {fold} — train: {len(train_df):,}  val: {len(val_df):,}")
    return train_df, val_df