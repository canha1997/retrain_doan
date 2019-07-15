
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf



FLAGS = None

#co tap tin hinh anh toi thieu
IMAGES_TOIDA = 2 ** 27 - 1  # ~134M
do_distort_images=0;

def images_data_list(image_dir, testing_data_phantram, validation_data_phantram):
  if not gfile.Exists(image_dir):
    tf.logging.error(" Directory cua images data '" + image_dir + "' khong duoc tim thay.")
    return None
  result = collections.OrderedDict()
  sub_dirs = [
    os.path.join(image_dir,item)
    for item in gfile.ListDirectory(image_dir)]
  sub_dirs = sorted(item for item in sub_dirs
                    if gfile.IsDirectory(item))
  for sub_dir in sub_dirs:
    dinh_dang_hinhanh = ['jpg', 'jpeg', 'JPG', 'JPEG']  #cac dinh dang tap hinh anh ma python co the decode
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    tf.logging.info("tim kiem hinh anh trong '" + dir_name + "'")
    for extension in dinh_dang_hinhanh:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      tf.logging.warning('Khong tim thay hinh anh nao')
      continue
    if len(file_list) < 20:
      tf.logging.warning(
          'CANHH BAO: Folder CO IT HON 20 TAM ANH, NEN XEM LAI TAP DATA.')
    elif len(file_list) > IMAGES_TOIDA:
      tf.logging.warning(
          'CANH BAO: Folder {} CO NHIEU HON {} TAM ANH. MOT SO HINH ANH SE '
          'SE KHONG BAO GIO DUOC CHON.'.format(dir_name, IMAGES_TOIDA))
    ten_label = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    data_training = []
    data_testing = []
    data_valdiation = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      #Phan chia tap anh cua chung ta thanh (valdiation + testing) va training
      #do chung ta de chung tat ca database vao chung 1 folder, voi thu viên hash
      #thi viec phan chia hop li cho model se duoc thuc hien
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      phantram_chia_nho = ((int(hash_name_hashed, 16) %
                          (IMAGES_TOIDA + 1)) *
                         (100.0 / IMAGES_TOIDA))
      if phantram_chia_nho < validation_data_phantram:
        data_valdiation.append(base_name)
      elif phantram_chia_nho < (testing_data_phantram + validation_data_phantram):
        data_testing.append(base_name)
      else:
        data_training.append(base_name)
    result[ten_label] = {
        'dir': dir_name,
        'training': data_training,
        'testing': data_testing,
        'validation': data_valdiation,
    }
  return result

  #lay path cua nhung hinh anh da duoc crop khuon matbao gom nhan~ cua no
   #Angry,happy,calm
def get_image_duongdan(lists_hinhanh, ten_label, index, image_dir, category):

  if ten_label not in lists_hinhanh:
    tf.logging.fatal('Labels khong he ton tai %s.', ten_label)
  danhsach_labels = lists_hinhanh[ten_label]
  if category not in danhsach_labels:
    tf.logging.fatal('Category Khong ton tai %s.', category)
  danhsach_category = danhsach_labels[category]
  if not danhsach_category:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     ten_label, category)
  mod_index = index % len(danhsach_category)
  base_name = danhsach_category[mod_index]
  sub_dir = danhsach_labels['dir']
  duongdan_daydu = os.path.join(image_dir, sub_dir, base_name)
  return duongdan_daydu

  #Lay duong link file tmp de luu tru nhung file txt
def lay_bottleneck_duongdan(lists_hinhanh, ten_label, index, bottleneck_dir,
                        category, architecture):         

  return get_image_duongdan(lists_hinhanh, ten_label, index, bottleneck_dir,
                        category) + '_' + architecture + '.txt'

#tao mot grap theo Tensorflow de luu tru mo hinh
def khoitao_model_graph(model_info):

  with tf.Graph().as_default() as graph:
    duongdan_model = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
    with gfile.FastGFile(duongdan_model, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tensor_cua_bottleneck, input_tensor_resize = (tf.import_graph_def(
          graph_def,
          name='',
          return_elements=[
              model_info['tensor_cua_bottleneck_name'],
              model_info['input_tensor_resize_name'],
          ]))
  return graph, tensor_cua_bottleneck, input_tensor_resize

#Resize lai input dau vao cua hinh anh
def bien_doi_hinhanh_jpg(input_width, input_height, input_depth, input_mean,
                      input_std):

  jpeg_hinhanh_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decode_hinhanh_tensor = tf.image.decode_jpeg(jpeg_hinhanh_data, channels=input_depth)
  #Bien doi input thanh tensor unit8 (con so interger 8 bit) voi x-D (x la depth cua buc anh)
  decode_hinhanh_tensor_as_float = tf.cast(decode_hinhanh_tensor, dtype=tf.float32)
  #Chinh vi ly do trong mobileV1 chi chay duoc voi du lieu tensor float
  #bien doi tensor thanh float
  decode_hinhanh_tensor_4d = tf.expand_dims(decode_hinhanh_tensor_as_float, 0)
  #add them 1 vao shape, vd mot matrix la [3,4]=>expand_dims=>[1,3,4]
  resize_hinhanh_shape = tf.stack([input_height, input_width])
  resize_hinhanh_shape_as_int = tf.cast(resize_hinhanh_shape, dtype=tf.int32)
  resize_hinhanh = tf.image.resize_bilinear(decode_hinhanh_tensor_4d,
                                           resize_hinhanh_shape_as_int)
  offset_image = tf.subtract(resize_hinhanh, input_mean)
  multi_hinhanh = tf.multiply(offset_image, 1.0 / input_std)
  return jpeg_hinhanh_data, multi_hinhanh
#Resize va decode hinh anh thanh cac network duoi dang thon tin tesnsor de dua vao network
def chay_bottleneck_decode_hinhanh(sess, image_data, image_data_tensor,
                            decode_hinhanh_tensor_tensor, input_tensor_resize,
                            tensor_cua_bottleneck):

 #resie image va rescale va decode no thong qua network
  resize_input_hinhanh = sess.run(decode_hinhanh_tensor_tensor,
                                  {image_data_tensor: image_data})
  # cho no qua network
  values_bottleneck = sess.run(tensor_cua_bottleneck,
                               {input_tensor_resize: resize_input_hinhanh})
  values_bottleneck = np.squeeze(values_bottleneck)
  return values_bottleneck




#dam bao rang o dia chua file image co ton tai
def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


bottleneck_path_2_values_bottleneck = {}

#tao mot file de chua nhung thong tin file bottleneck, mac dinh se la folder bottleneck layer trong file tmp
def create_bottleneck_file(bottleneck_path, lists_hinhanh, ten_label, index,
                           image_dir, category, sess, jpeg_hinhanh_data_tensor,
                           decode_hinhanh_tensor_tensor, input_tensor_resize,
                           tensor_cua_bottleneck):
  """Create a single bottleneck file."""
  tf.logging.info('Khoi tao file bottleneck tai ' + bottleneck_path)
  image_path = get_image_duongdan(lists_hinhanh, ten_label, index,
                              image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  image_data = gfile.FastGFile(image_path, 'rb').read()
  try:
    values_bottleneck = chay_bottleneck_decode_hinhanh(
        sess, image_data, jpeg_hinhanh_data_tensor, decode_hinhanh_tensor_tensor,
        input_tensor_resize, tensor_cua_bottleneck)
  except Exception as e:
    raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                 str(e)))
  bottleneck_string = ','.join(str(x) for x in values_bottleneck)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)

#Luu tru nhung bottleneck value vao trong file txt, neu khong ton tai file txt do thi
# lap tuc chuyen sang hinh anh khac
def get_or_create_bottleneck(sess, lists_hinhanh, ten_label, index, image_dir,
                             category, bottleneck_dir, jpeg_hinhanh_data_tensor,
                             decode_hinhanh_tensor_tensor, input_tensor_resize,
                             tensor_cua_bottleneck, architecture):

  danhsach_labels = lists_hinhanh[ten_label]
  sub_dir = danhsach_labels['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = lay_bottleneck_duongdan(lists_hinhanh, ten_label, index,
                                        bottleneck_dir, category, architecture)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, lists_hinhanh, ten_label, index,
                           image_dir, category, sess, jpeg_hinhanh_data_tensor,
                           decode_hinhanh_tensor_tensor, input_tensor_resize,
                           tensor_cua_bottleneck)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    values_bottleneck = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    tf.logging.warning('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, lists_hinhanh, ten_label, index,
                           image_dir, category, sess, jpeg_hinhanh_data_tensor,
                           decode_hinhanh_tensor_tensor, input_tensor_resize,
                           tensor_cua_bottleneck)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    values_bottleneck = [float(x) for x in bottleneck_string.split(',')]
  return values_bottleneck

#Dam bao rang cac du lieu bottleneck value, gom trainning, testing va valdiation
#da duoc luu tru
def cache_bottlenecks(sess, lists_hinhanh, image_dir, bottleneck_dir,
                      jpeg_hinhanh_data_tensor, decode_hinhanh_tensor_tensor,
                      input_tensor_resize, tensor_cua_bottleneck, architecture):

  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for ten_label, danhsach_labels in lists_hinhanh.items():
    for category in ['training', 'testing', 'validation']:
      danhsach_category = danhsach_labels[category]
      for index, unused_base_name in enumerate(danhsach_category):
        get_or_create_bottleneck(
            sess, lists_hinhanh, ten_label, index, image_dir, category,
            bottleneck_dir, jpeg_hinhanh_data_tensor, decode_hinhanh_tensor_tensor,
            input_tensor_resize, tensor_cua_bottleneck, architecture)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          tf.logging.info(
              str(how_many_bottlenecks) + ' bottleneck files created.')

#lay gia tri bottleneck tu folder tmp 
def get_random_cached_bottlenecks(sess, lists_hinhanh, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_hinhanh_data_tensor,
                                  decode_hinhanh_tensor_tensor, input_tensor_resize,
                                  tensor_cua_bottleneck, architecture):
  class_count = len(lists_hinhanh.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    #neu mot tam anh bi bien dang, thi lap tuc tam anh do se bi bo qua, vi mot khi tam anh bi bien dang thi ko the lay bottleneck
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      ten_label = list(lists_hinhanh.keys())[label_index]
      image_index = random.randrange(IMAGES_TOIDA + 1)
      image_name = get_image_duongdan(lists_hinhanh, ten_label, image_index,
                                  image_dir, category)
      bottleneck = get_or_create_bottleneck(
          sess, lists_hinhanh, ten_label, image_index, image_dir, category,
          bottleneck_dir, jpeg_hinhanh_data_tensor, decode_hinhanh_tensor_tensor,
          input_tensor_resize, tensor_cua_bottleneck, architecture)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Lay tat ca gia tri bottleneck 
    for label_index, ten_label in enumerate(lists_hinhanh.keys()):
      for image_index, image_name in enumerate(
          lists_hinhanh[ten_label][category]):
        image_name = get_image_duongdan(lists_hinhanh, ten_label, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(
            sess, lists_hinhanh, ten_label, image_index, image_dir, category,
            bottleneck_dir, jpeg_hinhanh_data_tensor, decode_hinhanh_tensor_tensor,
            input_tensor_resize, tensor_cua_bottleneck, architecture)
        #thiet lap dau ra thuc te voi moi index
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0    #dau ra thuc te se lay index =1 tuc \
        #100%
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames





def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

#add gia tri bottleneck vao bottleneck layer va bat dau training
#Ta dung ham softmax function de dua gia tri output ve dang sac xuat
def add_final_training_ops(class_count, final_tensor_name, tensor_cua_bottleneck,
                           tensor_cua_bottleneck_size):

  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        tensor_cua_bottleneck,
        shape=[None, tensor_cua_bottleneck_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal(
          [tensor_cua_bottleneck_size, class_count], stddev=0.001)

      layer_weights = tf.Variable(initial_value, name='final_weights')

      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)
#Dung softmax function tai layer cuoi cung, tuc output (final tensor)
  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)

#Buoc thuc hien danh gia model
def add_evaluation_step(result_tensor, ground_truth_tensor):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      ground_true=tf.argmax(ground_truth_tensor, 1)
      #ham equal tuc tra ve True hay False khi so sanh tung phan tu giua x va y co gan bang nhau hay khong
      #ham argmax: tra ve gia tri lon nhat trong vector => tuc la label
      #=> Ham equal dung de so sanh giua gia tri ground-true(thuc te) va gia tri dau ra cua ta
      #tu do tinh do chinh xac
      correct_prediction = tf.equal(
          prediction, tf.argmax(ground_truth_tensor, 1))
      #danh gia model
      #trong do tf.cast bien doi ve 32 bit(tf.float32) maqual lai chi tra ve True-False nen cast se bien
      #doi la 0 hoac 1
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction, ground_true

#Luu tru grap vo mot folder, khi ta run no se lay grap nay va dua len session
def save_graph_to_file(sess, graph, graph_file_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return
def prepare_file_system():
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if FLAGS.intermediate_store_frequency > 0:
    ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
  return


def create_model_info(architecture):
  """Given the name of a model architecture, returns information about it.

  There are different base image recognition pretrained models that can be
  retrained using transfer learning, and this function translates from the name
  of a model to the attributes that are needed to download and train with it.

  Args:
    architecture: Name of a model architecture.

  Returns:
    Dictionary of information about the model, or None if the name isn't
    recognized

  Raises:
    ValueError: If architecture name is unknown.
  """
  architecture = architecture.lower()
  if architecture == 'inception_v3':
    # pylint: disable=line-too-long
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    # pylint: enable=line-too-long
    tensor_cua_bottleneck_name = 'pool_3/_reshape:0'
    tensor_cua_bottleneck_size = 2048
    input_width = 299
    input_height = 299
    input_depth = 3
    input_tensor_resize_name = 'Mul:0'
    model_file_name = 'classify_image_graph_def.pb'
    input_mean = 128
    input_std = 128
  elif architecture.startswith('mobilenet_'):
    parts = architecture.split('_')
    if len(parts) != 3 and len(parts) != 4:
      tf.logging.error("Couldn't understand architecture name '%s'",
                       architecture)
      return None
    version_string = parts[1]
    if (version_string != '1.0' and version_string != '0.75' and
        version_string != '0.50' and version_string != '0.25'):
      tf.logging.error(
          """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
  but found '%s' for architecture '%s'""",
          version_string, architecture)
      return None
    size_string = parts[2]
    if (size_string != '224' and size_string != '192' and
        size_string != '160' and size_string != '128'):
      tf.logging.error(
          """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""",
          size_string, architecture)
      return None
    if len(parts) == 3:
      is_quantized = False
    else:
      if parts[3] != 'quantized':
        tf.logging.error(
            "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
            architecture)
        return None
      is_quantized = True
    data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
    data_url += version_string + '_' + size_string + '_frozen.tgz'
    tensor_cua_bottleneck_name = 'MobilenetV1/Predictions/Reshape:0'
    tensor_cua_bottleneck_size = 1001
    input_width = int(size_string)
    input_height = int(size_string)
    input_depth = 3
    input_tensor_resize_name = 'input:0'
    if is_quantized:
      model_base_name = 'quantized_graph.pb'
    else:
      model_base_name = 'frozen_graph.pb'
    model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
    model_file_name = os.path.join(model_dir_name, model_base_name)
    input_mean = 127.5
    input_std = 127.5
  else:
    tf.logging.error("Couldn't understand architecture name '%s'", architecture)
    raise ValueError('Unknown architecture', architecture)

  return {
      'data_url': data_url,
      'tensor_cua_bottleneck_name': tensor_cua_bottleneck_name,
      'tensor_cua_bottleneck_size': tensor_cua_bottleneck_size,
      'input_width': input_width,
      'input_height': input_height,
      'input_depth': input_depth,
      'input_tensor_resize_name': input_tensor_resize_name,
      'model_file_name': model_file_name,
      'input_mean': input_mean,
      'input_std': input_std,
  }





def main(_):
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # Prepare necessary directories  that can be used during training
  prepare_file_system()

  # Gather information about the model architecture we'll be using.
  model_info = create_model_info(FLAGS.architecture)
  if not model_info:
    tf.logging.error('Did not recognize architecture flag')
    return -1

  # Set up the pre-trained graph.
 
  graph, tensor_cua_bottleneck, resize_hinhanh_tensor = (
      khoitao_model_graph(model_info))

  # Look at the folder structure, and create lists of all the images.
  lists_hinhanh = images_data_list(FLAGS.image_dir, FLAGS.testing_data_phantram,
                                   FLAGS.validation_data_phantram)
  class_count = len(lists_hinhanh.keys())
  if class_count == 0:
    tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  if class_count == 1:
    tf.logging.error('Only one valid folder of images found at ' +
                     FLAGS.image_dir +
                     ' - multiple classes are needed for classification.')
    return -1



  with tf.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph.
    jpeg_hinhanh_data_tensor, decode_hinhanh_tensor_tensor = bien_doi_hinhanh_jpg(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

    if do_distort_images:
      # We will be applying distortions, so setup the operations we'll need.
      (distorted_jpeg_hinhanh_data_tensor,
       distorted_image_tensor) = add_input_distortions(
           FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
           FLAGS.random_brightness, model_info['input_width'],
           model_info['input_height'], model_info['input_depth'],
           model_info['input_mean'], model_info['input_std'])
    else:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      cache_bottlenecks(sess, lists_hinhanh, FLAGS.image_dir,
                        FLAGS.bottleneck_dir, jpeg_hinhanh_data_tensor,
                        decode_hinhanh_tensor_tensor, resize_hinhanh_tensor,
                        tensor_cua_bottleneck, FLAGS.architecture)

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(
         len(lists_hinhanh.keys()), FLAGS.final_tensor_name, tensor_cua_bottleneck,
         model_info['tensor_cua_bottleneck_size'])

    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, prediction, ground_true = add_evaluation_step(
        final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)
   
    # Run the training for as many cycles as requested on the command line.
    for i in range(FLAGS.how_many_training_steps):
      # Get a batch of input bottleneck values, either calculated fresh every
      # time with distortions applied, or from the cache stored on disk.
      if do_distort_images:
        (train_bottlenecks,
         train_ground_truth) = get_random_distorted_bottlenecks(
             sess, lists_hinhanh, FLAGS.train_batch_size, 'training',
             FLAGS.image_dir, distorted_jpeg_hinhanh_data_tensor,
             distorted_image_tensor, resize_hinhanh_tensor, tensor_cua_bottleneck)
      else:
        (train_bottlenecks,
         train_ground_truth, _) = get_random_cached_bottlenecks(
             sess, lists_hinhanh, FLAGS.train_batch_size, 'training',
             FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_hinhanh_data_tensor,
             decode_hinhanh_tensor_tensor, resize_hinhanh_tensor, tensor_cua_bottleneck,
             FLAGS.architecture)
      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.
      train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == FLAGS.how_many_training_steps)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                        (datetime.now(), i, train_accuracy * 100))
        tf.logging.info('%s: Step %d: Cross entropy = %f' %
                        (datetime.now(), i, cross_entropy_value))
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, lists_hinhanh, FLAGS.validation_batch_size, 'validation',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_hinhanh_data_tensor,
                decode_hinhanh_tensor_tensor, resize_hinhanh_tensor, tensor_cua_bottleneck,
                FLAGS.architecture))
              
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (datetime.now(), i, validation_accuracy * 100,
                         len(validation_bottlenecks)))
       
      # Store intermediate results
      intermediate_frequency = FLAGS.intermediate_store_frequency

      if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
          and i > 0):
        intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                  'intermediate_' + str(i) + '.pb')
        tf.logging.info('Save intermediate result to : ' +
                        intermediate_file_name)
        save_graph_to_file(sess, graph, intermediate_file_name)

    #Trainning da hoan thanh thi bat dau su dung file valdiation va testing de test do chinh xac
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
            sess, lists_hinhanh, FLAGS.test_batch_size, 'testing',
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_hinhanh_data_tensor,
            decode_hinhanh_tensor_tensor, resize_hinhanh_tensor, tensor_cua_bottleneck,
            FLAGS.architecture))
    test_accuracy, predictions, ground_trues = sess.run(
        [evaluation_step, prediction, ground_true],
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                    (test_accuracy * 100, len(test_bottlenecks)))
    print(" Day la output dau ra DU DOAN ")
    print(predictions)
    print("DAY LA OUTPUT CUA DAU RA THUC TE")
    print(ground_trues)
    if FLAGS.print_misclassified_test_images:
      tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():
          tf.logging.info('%70s  %s' %
                          (test_filename,
                           list(lists_hinhanh.keys())[predictions[i]]))

    # Write out the trained graph and labels with the weights stored as
    # constants.
    save_graph_to_file(sess, graph, FLAGS.output_graph)
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(lists_hinhanh.keys()) + '\n')
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/tmp/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=6000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_data_phantram',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_data_phantram',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
     
  )
  parser.add_argument(
      '--flip_left_right',
      default=False,
     
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
       help="""\
     
      """ 
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      
      """
    
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
       help="""\
     ]
      """
     )
  parser.add_argument(
      '--architecture',
      type=str,
      default='inception_v3',
       help="""\
     
      """)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
  
