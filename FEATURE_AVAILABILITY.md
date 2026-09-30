# Feature availability before departure

The public benchmark omits measurement timestamps and a prediction horizon. Treat these fields as retrospective
features until a company can establish when they were recorded. A “next 90 days” claim cannot be inferred here.

| Feature | Required pre-departure definition | Main ambiguity |
|---|---|---|
| satisfaction_level | Most recent survey available at scoring time | Survey could have occurred during exit processing |
| last_evaluation | Evaluation completed before the scoring snapshot | Evaluation may overlap with the departure decision |
| number_project | Projects assigned as of the snapshot | Retrospective counts can include later activity |
| average_montly_hours | Trailing hours from a fixed window ending before scoring | Window boundaries are undocumented |
| time_spend_company | Tenure at scoring, using a dated hire record | Could represent final tenure at departure |
| Work_accident | Accidents reported before scoring | Entire-employment aggregation can include future events |
| promotion_last_5years | Promotions in the five years before scoring | Window anchor is unknown |
| department, salary | Values effective at the snapshot | Transfers or changes may follow a resignation |

Required external validation data: anonymized employee ID, snapshot date, feature measurement windows,
departure event date, and an explicit horizon. Split by time and employee; apply eligibility rules before labels.
Feature importance is associative. A high-ranked feature does not establish why someone left or which intervention works.
