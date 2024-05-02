#hyperparameters

C, H, W = 3,112,112
input_resize = 171,128
test_batch_size = 1

# serialize the weights and biases of models using.pth

m1_path = None
m2_path = None 
m3_path = None
m4_path = None
c3d_path = None 

with_dive_classification = False
with_caption = False

max_epochs = 100

model_ckpt_interval = 1 #in epochs

base_learning_rate = 0.0001

temporal_stride = 16

BUCKET_NAME = 'aqa-diving'
BUCKET_WEIGHT_FC6 = 'model_my_fc6_94.pth'
BUCKET_WEIGHT_CNN = 'model_CNN_94.pth'