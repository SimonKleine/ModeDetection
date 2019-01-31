import accelerometerfeatures.utils.pytorch.dataset as dataset


dataset = dataset.AccelerometerDatasetLoader("/home/simonk/Documents/Bachelorarbeit/data/accelerometer_data/1st_study.csv")
'''print(dataset.get_user_data(("a526f3566e9c9024dfa7378eb4291d787a09fd37")))
for data_shred in dataset.get_user_data("a526f3566e9c9024dfa7378eb4291d787a09fd37"):
    if data_shred.empty:
        continue
    print("lap")
    first_idx = data_shred.first_valid_index()
    last_idx = data_shred.last_valid_index()
    last_datetime = data_shred.timestamp[last_idx]
    start_datetime = data_shred.timestamp[first_idx]
    #end_datetime = start_datetime + win_size
    print(first_idx)
    print(last_idx)
    print(last_datetime)
    print(start_datetime)
    #print(end_datetime)
'''
print(dataset.get_user_data_windows("a526f3566e9c9024dfa7378eb4291d787a09fd37"))