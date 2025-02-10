from representations import *
from gen_metrics import *
import pandas as pd
import numpy as np

datasets = ['MSL', 'SMAP', 'SMD', 'swat', 'GECCO', 'Swan', 'Yahoo', 'KPI', 'UCR']
injectors = ['genias', 'couta', 'ncad', 'cutaddpaste', 'carla']

for dataset in datasets:
    print(dataset)
    all_files = os.listdir(os.path.join('generated_anoms', dataset))
    all_anomalous = {}
    first_row = True
    for entity in all_files:
        injections_path = os.path.join('generated_anoms', dataset, entity)
        for injector in injectors:
            all_anomalous[injector] = torch.load(os.path.join(injections_path, f"{injector}_rep.pth"))
        train_test_rep = torch.load(os.path.join(injections_path, "train_test_rep.pth"))
        test_representations = train_test_rep["test_representations"]
        test_labels = train_test_rep["test_labels"]

        diversity_score = get_EDI_scores(all_anomalous, 2)
        ARP_score = get_ARP_scores(all_anomalous, test_representations, test_labels)

        if first_row:
            results_df = pd.DataFrame([{'entity': entity} | diversity_score | realism_score])
            first_row = False
        else:
            row = pd.DataFrame([{'entity': entity} | diversity_score | realism_score])
            results_df = pd.concat([results_df, row], ignore_index=True)

    avg_values = results_df.iloc[:, 1:].mean()
    avg_row = pd.DataFrame([["avg"] + avg_values.tolist()], columns=results_df.columns)
    results_df = pd.concat([results_df, avg_row], ignore_index=True)
    results_df.to_csv(f'generated_anoms/{dataset}_results.csv')