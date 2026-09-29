import numpy as np
import pandas as pd
import pytest
from analysis import NUMERIC, load_data, prepare_features, preprocessing


def example():
    return pd.DataFrame({"satisfaction_level":[0.2,0.8]*20, "last_evaluation":[0.4,0.9]*20,
        "number_project":np.arange(40)%5+1, "average_montly_hours":np.arange(40)+150,
        "time_spend_company":[3]*40, "Work_accident":[0]*40, "promotion_last_5years":[0]*40,
        "sales":["sales","technical"]*20, "salary":["low","high"]*20, "left":[0,1]*20})


def test_duplicates_removed_before_splitting(tmp_path):
    frame = example()
    frame = pd.concat([frame, frame.iloc[:10]], ignore_index=True)
    p = tmp_path/"hr.csv"
    frame.to_csv(p,index=False)
    X,y,info = load_data(p)
    assert len(X)==40 and len(y)==40
    assert info["duplicates_removed"]==10
    assert "left" not in X


def test_train_only_scaling_and_unknown_department_supported():
    X=prepare_features(example())
    pipeline=preprocessing().fit(X.iloc[:30])
    transformed=pipeline.transform(X.iloc[30:].assign(department="new_department"))
    assert np.isfinite(transformed).all()
    scaler=pipeline.named_transformers_["numeric"].named_steps["standardscaler"]
    assert scaler.mean_[NUMERIC.index("average_montly_hours")]==pytest.approx(X.iloc[:30].average_montly_hours.mean())


def test_impossible_satisfaction_is_rejected():
    frame=example()
    frame.loc[0,"satisfaction_level"]=4.0
    with pytest.raises(ValueError,match="between 0 and 1"):
        prepare_features(frame)
