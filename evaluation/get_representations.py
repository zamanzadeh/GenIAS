from representations import *

datasets = ['MSL', 'SMAP', 'SMD', 'swat', 'GECCO', 'Swan', 'Yahoo', 'KPI', 'UCR']
injectors = ['genias', 'couta', 'ncad', 'cutaddpaste', 'carla']

for dataset in datasets:
    if (dataset == 'MSL') | (dataset == 'SMAP'):
        window_size=200
        step_size=1
        with open('../../datasets/MSL_SMAP/labeled_anomalies.csv', 'r') as file:
            csv_reader = pd.read_csv(file, delimiter=',')
        data_info = csv_reader[csv_reader['spacecraft'] == dataset]
        file_list = data_info['chan_id']
        file_list = sorted(file_list)

        for filename in file_list:
            print(f">>======= Dataset: {dataset} | Entity: {filename} =======>>")
            for injector in injectors:
                if injector == 'genias':
                    injector_params={'gen_model':f'{dataset.lower()}_kl12', 'scale': 1.0, 'tau': 0.05}
                elif injector == 'couta':
                    injector_params=None
                elif injector == 'ncad':
                    injector_params={'coe_rate': 1.0, 'mixup_rate': 1.0, 'suspect_win_len': 10}
                elif injector == 'cutaddpaste':
                    injector_params={'cut_rate': 16, 'dim': 15,  'trend_rate': 0.1}
                elif injector == 'carla':
                    injector_params = None
                rep, scores = get_representation(injector=injector, dataset=dataset, entity=filename, window_size=window_size, step_size=step_size, with_training=False, injector_params=injector_params, save=True)
                    
    elif dataset == 'SMD':
        window_size=200
        step_size=20
        all_files = os.listdir(os.path.join('../../datasets', 'SMD/train'))
        files = [file for file in all_files if file.startswith('machine-')]
        file_list = sorted(files)

        for filename in file_list:
            print(f">>======= Dataset: {dataset} | Entity: {filename} =======>>")
            for injector in injectors:
                if injector == 'genias':
                    injector_params={'gen_model':f'{dataset.lower()}_kl12', 'scale': 1.0, 'tau': 0.01}
                elif injector == 'couta':
                    injector_params=None
                elif injector == 'ncad':
                    injector_params={'coe_rate': 1.0, 'mixup_rate': 1.0, 'suspect_win_len': 10}
                elif injector == 'cutaddpaste':
                    injector_params={'cut_rate': 16, 'dim': 15,  'trend_rate': 0.1}
                elif injector == 'carla':
                    injector_params = None
                rep, scores = get_representation(injector=injector, dataset=dataset, entity=filename, window_size=window_size, step_size=step_size, with_training=False, injector_params=injector_params, save=True)

    elif dataset == 'swat':
        entity = 'swat'
        window_size=200
        step_size=20
        print(f">>======= Dataset: {dataset} | Entity: {dataset} =======>>")
        for injector in injectors:
            if injector == 'genias':
                injector_params={'gen_model':f'{dataset.lower()}_kl12', 'scale': 1.0, 'tau': 0.01}
            elif injector == 'couta':
                injector_params=None
            elif injector == 'ncad':
                injector_params={'coe_rate': 1.0, 'mixup_rate': 1.0, 'suspect_win_len': 10}
            elif injector == 'cutaddpaste':
                injector_params={'cut_rate': 16, 'dim': 15,  'trend_rate': 0.1}
            elif injector == 'carla':
                injector_params = None
            rep, scores = get_representation(injector=injector, dataset=dataset, entity=entity, window_size=window_size, step_size=step_size, with_training=False, injector_params=injector_params, save=True)
    
    elif dataset == 'GECCO':
        entity='GECCO'
        window_size=200
        step_size=100
        print(f">>======= Dataset: {dataset} | Entity: {dataset} =======>>")
        for injector in injectors:
            if injector == 'genias':
                injector_params={'gen_model':f'{dataset.lower()}_kl12', 'scale': 1.0, 'tau': 0.01}
            elif injector == 'couta':
                injector_params=None
            elif injector == 'ncad':
                injector_params={'coe_rate': 1.0, 'mixup_rate': 1.0, 'suspect_win_len': 10}
            elif injector == 'cutaddpaste':
                injector_params={'cut_rate': 16, 'dim': 15,  'trend_rate': 0.1}
            elif injector == 'carla':
                injector_params = None
            rep, scores = get_representation(injector=injector, dataset=dataset, entity=entity, window_size=window_size, step_size=step_size, with_training=False, injector_params=injector_params, save=True)

    elif dataset == 'swan':
        entity = 'swan'
        window_size=200
        step_size=100
        print(f">>======= Dataset: {dataset} | Entity: {dataset} =======>>")
        for injector in injectors:
            if injector == 'genias':
                injector_params={'gen_model':f'{dataset.lower()}_kl12', 'scale': 1.0, 'tau': 0.01}
            elif injector == 'couta':
                injector_params=None
            elif injector == 'ncad':
                injector_params={'coe_rate': 1.0, 'mixup_rate': 1.0, 'suspect_win_len': 10}
            elif injector == 'cutaddpaste':
                injector_params={'cut_rate': 16, 'dim': 15,  'trend_rate': 0.1}
            elif injector == 'carla':
                injector_params = None
            rep, scores = get_representation(injector=injector, dataset=dataset, entity=entity, window_size=window_size, step_size=step_size, with_training=False, injector_params=injector_params, save=True)

    elif dataset == 'Yahoo':
        window_size=200
        step_size=1
        all_files = os.listdir(os.path.join('../../datasets', 'Yahoo'))
        files = [file for file in all_files if file.startswith('real')]
        file_list = sorted(files)

        for filename in file_list:
            print(f">>======= Dataset: {dataset} | Entity: {filename} =======>>")
            for injector in injectors:
                if injector == 'genias':
                    injector_params={'gen_model':f'{dataset.lower()}_kl12', 'scale': 1.0, 'tau': 0.01}
                elif injector == 'couta':
                    injector_params=None
                elif injector == 'ncad':
                    injector_params={'coe_rate': 1.0, 'mixup_rate': 1.0, 'suspect_win_len': 10}
                elif injector == 'cutaddpaste':
                    injector_params={'cut_rate': 16, 'dim': 15,  'trend_rate': 0.1}
                elif injector == 'carla':
                    injector_params = None
                rep, scores = get_representation(injector=injector, dataset=dataset, entity=filename, window_size=window_size, step_size=step_size, with_training=False, injector_params=injector_params, save=True)

    elif dataset == 'KPI':
        window_size=200
        step_size=10
        all_files = os.listdir(os.path.join('../../datasets', 'KPI/train'))
        file_list = sorted(all_files)

        for filename in file_list:
            print(f">>======= Dataset: {dataset} | Entity: {filename} =======>>")
            for injector in injectors:
                if injector == 'genias':
                    injector_params={'gen_model':f'{dataset.lower()}_kl12', 'scale': 1.0, 'tau': 0.01}
                elif injector == 'couta':
                    injector_params=None
                elif injector == 'ncad':
                    injector_params={'coe_rate': 1.0, 'mixup_rate': 1.0, 'suspect_win_len': 10}
                elif injector == 'cutaddpaste':
                    injector_params={'cut_rate': 16, 'dim': 15,  'trend_rate': 0.1}
                elif injector == 'carla':
                    injector_params = None
                rep, scores = get_representation(injector=injector, dataset=dataset, entity=filename, window_size=window_size, step_size=step_size, with_training=False, injector_params=injector_params, save=True)

    elif dataset == 'UCR':
        window_size=200
        step_size=10
        all_files = os.listdir(os.path.join('../../datasets', 'UCR'))
        file_list = sorted(all_files)

        for filename in file_list:
            print(f">>======= Dataset: {dataset} | Entity: {filename} =======>>")
            for injector in injectors:
                if injector == 'genias':
                    injector_params={'gen_model':f'{dataset.lower()}_kl12', 'scale': 1.0, 'tau': 0.01}
                elif injector == 'couta':
                    injector_params=None
                elif injector == 'ncad':
                    injector_params={'coe_rate': 1.0, 'mixup_rate': 1.0, 'suspect_win_len': 10}
                elif injector == 'cutaddpaste':
                    injector_params={'cut_rate': 16, 'dim': 15,  'trend_rate': 0.1}
                elif injector == 'carla':
                    injector_params = None
                rep, scores = get_representation(injector=injector, dataset=dataset, entity=filename, window_size=window_size, step_size=step_size, with_training=False, injector_params=injector_params, save=True)
