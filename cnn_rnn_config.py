classes_ = '_ !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


cdict_ = {c:i for i,c in enumerate(classes_)}
icdict_ = {i:c for i,c in enumerate(classes_)}

k = 1
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]

head_cfg = (256, 3)  # (hidden , num_layers)

head_type = 'rnn'

flattening = 'maxpool'
fixed_size = (4 * 32,  4*  256)
#flattening='concat'


# Spatial Transformer Network
stn = False

max_epochs = 140
batch_size = 10
early_stopping = 3

save_path = './saved_models/cnn_rnn/'
load_model = False