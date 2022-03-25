import onnx
from onnx_tf.backend import prepare

# method 1, cmd:
# onnx-tf convert -i ./mnist.onnx -o ./mnist_savedmodel_dir
# method 2:
onnx_model = onnx.load("mnist.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("./mnist_savedmodel_dir_2")


# tensorflow and tensorflow-addons versions: https://github.com/tensorflow/addons
# tensorflow and tensorflow-probability versions: https://github.com/tensorflow/probability/releases


# $ saved_model_cli show --dir mnist_savedmodel_dir/ --all
# $ saved_model_cli show --dir mnist_savedmodel_dir_2/ --all
# 2022-03-25 15:17:09.351237: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
# 2022-03-25 15:17:09.351288: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
#
# MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
#
# signature_def['__saved_model_init_op']:
#   The given SavedModel SignatureDef contains the following input(s):
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['__saved_model_init_op'] tensor_info:
#         dtype: DT_INVALID
#         shape: unknown_rank
#         name: NoOp
#   Method name is:
#
# signature_def['serving_default']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['input.1'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (1, 1, 28, 28)
#         name: serving_default_input.1:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['20'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (1, 10)
#         name: PartitionedCall:0
#   Method name is: tensorflow/serving/predict
#
# Defined Functions:
#   Function Name: '__call__'
#         Named Argument #1
#           input.1
#
#   Function Name: 'gen_tensor_dict'
