
python score_brat.py \
    ../dataset/GenQA/predictions/llama80b_events/ref \
    ../dataset/GenQA/predictions/llama80b_events/pred \
    ../dataset/GenQA/predictions/llama80b_events/score_ \
    --score_trig overlap  \
    --score_span overlap  \
    --score_labeled label