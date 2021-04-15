import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
  '--file', help='File path of .tflite file.', required=True)
args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model(args.file)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():
  for _ in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    yield [input]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8

tflite_quant_model = converter.convert()