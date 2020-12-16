#-*-coding:utf-8-*-

import sys
sys.path.append('.')
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
import cv2
import tqdm
from my_logger import Logger
from tensorpack.dataflow import DataFromGenerator
from  lib.dataset.dataietr import DataIter, DsfdDataIter
from lib.core.model.net.ssd import DSFD
from train_config import config as cfg
from train_config import token


from lib.BoundingBox import *
from lib.BoundingBoxes import *
from lib.Evaluator import *

import numpy as np
from mean_average_precision import MetricBuilder

metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)


import neptune

neptune.init(project_qualified_name='rizhiy/kkk',
             api_token=token,
             )

from lib.helper.logger import logger

class trainner():
    def __init__(self):
        self.train_ds = DataIter(cfg.DATA.root_path,cfg.DATA.train_txt_path,training_flag=True)
        self.val_ds = DataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,training_flag=False)

        self.val_metric_gen = DsfdDataIter(cfg.DATA.root_path,cfg.DATA.val_txt_path,training_flag=False)
        self.train_metric_gen = DsfdDataIter(cfg.DATA.root_path,cfg.DATA.train_txt_path,training_flag=False)
        self.trainds_for_eval =  DataIter(cfg.DATA.root_path,cfg.DATA.train_txt_path,training_flag=False)

        self.inputs=[]
        self.outputs=[]
        self.val_outputs=[]
        self.ite_num=1

        self.ev = Evaluator()

        self._graph = tf.Graph()



        self.summaries=[]

        self.ema_weights=False



    def get_opt(self):

        with self._graph.as_default():
            ##set the opt there
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), dtype=tf.int32, trainable=False)

            # Decay the learning rate
            lr = tf.train.piecewise_constant(global_step,
                                             cfg.TRAIN.lr_decay_every_step,
                                             cfg.TRAIN.lr_value_every_step
                                             )
            opt = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=False)
            return opt,lr,global_step

    def load_weight(self):

        with self._graph.as_default():

            if cfg.MODEL.continue_train:
                #########################restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                for v in tf.global_variables():
                    if 'moving_mean' in v.name or 'moving_variance' in v.name:
                            variables_restore.append(v)
                saver2 = tf.train.Saver(variables_restore)
                saver2.restore(self.sess, cfg.MODEL.pretrained_model)

            elif cfg.MODEL.pretrained_model is not None:
                #########################restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=cfg.MODEL.net_structure)

                for v in tf.global_variables():
                    if 'moving_mean' in v.name or 'moving_variance' in v.name:
                        if cfg.MODEL.net_structure in v.name:
                            variables_restore.append(v)
                print(variables_restore)

                variables_restore_n = [v for v in variables_restore if
                                       'GN' not in v.name]
                                      # and 'BatchNorm/beta' not in v.name and 'BatchNorm/gamma' not in v.name
                                      #  and 'BatchNorm/moving_mean' not in v.name and 'BatchNorm/moving_variance' not in v.name]  # Conv2d_1c_1x1 Bottleneck
                print(variables_restore_n)
                saver2 = tf.train.Saver(variables_restore_n)
                saver2.restore(self.sess, cfg.MODEL.pretrained_model)
            else:
                logger.info('no pretrained model, train from sctrach')
                # Build an initialization operation to run below.
    def frozen(self):
        with self._graph.as_default():



            variables_need_grads=[]


            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


            for v in variables:
                training_flag = True
                if cfg.TRAIN.frozen_stages>=0:

                    if '%s/conv1'%cfg.MODEL.net_structure in v.name:
                        training_flag=False

                for i in range(1,1+cfg.TRAIN.frozen_stages):
                    if '%s/block%d'%(cfg.MODEL.net_structure,i) in v.name:
                        training_flag=False
                        break

                if training_flag:
                    variables_need_grads.append(v)
                else:
                    v_stop= tf.stop_gradient(v)
            return variables_need_grads

    def add_summary(self,event):
        self.summaries.append(event)

    def tower_loss(self,scope, images, labels,boxes,L2_reg, training):
        """Calculate the total loss on a single tower running the model.

        Args:
          scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
          images: Images. 4D tensor of shape [batch_size, height, width, 3].
          labels: Labels. 1D tensor of shape [batch_size].

        Returns:
           Tensor of shape [] containing the total loss for a batch of data
        """

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        ssd=DSFD()
        reg_loss, cla_loss, result_network =ssd.forward(images,boxes,labels, L2_reg, training)

        #reg_loss,cla_loss=ssd_loss( reg, cla,boxes,labels)
        regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')

        return reg_loss,cla_loss,regularization_losses, result_network
    def average_gradients(self,tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """

        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                try:
                    g=tf.clip_by_value(g, -5., 5.)
                    expanded_g = tf.expand_dims(g, 0)
                except:
                    print(_)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    def build(self):

        with self._graph.as_default(), tf.device('/cpu:0'):

            # Create an optimizer that performs gradient descent.
            opt, lr, global_step = self.get_opt()

            ##some global placeholder
            L2_reg = tf.placeholder(tf.float32, name="L2_reg")
            training = tf.placeholder(tf.bool, name="training_flag")

            total_loss_to_show = 0.
            images_place_holder_list = []
            labels_place_holder_list = []
            boxes_place_holder_list = []

            weights_initializer = slim.xavier_initializer()
            biases_initializer = tf.constant_initializer(0.)
            biases_regularizer = tf.no_regularizer
            weights_regularizer = tf.contrib.layers.l2_regularizer(L2_reg)

            # Calculate the gradients for each model tower.
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(cfg.TRAIN.num_gpu):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % (i)) as scope:
                            with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

                                images_ = tf.placeholder(tf.float32, [None, None,None, 3], name="images")
                                boxes_ = tf.placeholder(tf.float32, [cfg.TRAIN.batch_size, None, 4], name="input_boxes")
                                labels_ = tf.placeholder(tf.int64, [cfg.TRAIN.batch_size, None], name="input_labels")
                                ###total anchor

                                images_place_holder_list.append(images_)
                                labels_place_holder_list.append(labels_)
                                boxes_place_holder_list.append(boxes_)

                                with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                                                     slim.conv2d_transpose, slim.separable_conv2d,
                                                     slim.fully_connected],
                                                    weights_regularizer=weights_regularizer,
                                                    biases_regularizer=biases_regularizer,
                                                    weights_initializer=weights_initializer,
                                                    biases_initializer=biases_initializer):
                                    reg_loss, cla_loss, l2_loss, result_network = self.tower_loss(
                                        scope, images_, labels_, boxes_, L2_reg, training)

                                    ##use muti gpu ,large batch
                                    if i == cfg.TRAIN.num_gpu - 1:
                                        total_loss = tf.add_n([reg_loss, cla_loss, l2_loss])
                                    else:
                                        total_loss = tf.add_n([reg_loss, cla_loss])
                                total_loss_to_show += total_loss
                                # Reuse variables for the next tower.
                                tf.get_variable_scope().reuse_variables()

                                ##when use batchnorm, updates operations only from the
                                ## final tower. Ideally, we should grab the updates from all towers
                                # but these stats accumulate extremely fast so we can ignore the
                                #  other stats from the other towers without significant detriment.
                                bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

                                # Retain the summaries from the final tower.
                                self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                                ###freeze some params
                                train_var_list = self.frozen()
                                # Calculate the gradients for the batch of data on this CIFAR tower.
                                grads = opt.compute_gradients(total_loss, train_var_list)

                                # Keep track of the gradients across all towers.
                                tower_grads.append(grads)
            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = self.average_gradients(tower_grads)

            # Add a summary to track the learning rate.
            self.add_summary(tf.summary.scalar('learning_rate', lr))
            self.add_summary(tf.summary.scalar('total_loss', total_loss_to_show))
            self.add_summary(tf.summary.scalar('loc_loss', reg_loss))
            self.add_summary(tf.summary.scalar('cla_loss', cla_loss))
            self.add_summary(tf.summary.scalar('l2_loss', l2_loss))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    self.add_summary(tf.summary.histogram(var.op.name + '/gradients', grad))

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                self.add_summary(tf.summary.histogram(var.op.name, var))

            if self.ema_weights:
                # Track the moving averages of all trainable variables.
                variable_averages = tf.train.ExponentialMovingAverage(
                    0.9, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                # Group all updates to into a single train op.
                train_op = tf.group(apply_gradient_op, variables_averages_op, *bn_update_ops)
            else:
                train_op = tf.group(apply_gradient_op, *bn_update_ops)



            ###set inputs and ouputs
            self.inputs = [images_place_holder_list,
                           boxes_place_holder_list,
                           labels_place_holder_list,
                           L2_reg,
                           training]
            self.outputs = [train_op,
                            total_loss_to_show,
                            reg_loss,
                            cla_loss,
                            l2_loss,
                            lr,
                            result_network]
            self.val_outputs = [total_loss_to_show,
                                reg_loss,
                                cla_loss,
                                l2_loss,
                                lr,
                                result_network]


            tf_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_config)

            ##init all variables
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.Logg = Logger(self.sess)
            ######
    def train_loop(self):
        """Train faces data for a number of epoch."""
        neptune.create_experiment(cfg.MODEL.model_path)

        self.build()
        self.load_weight()

        print("start train")



        with self._graph.as_default():
            # Create a saver.
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # Build the summary operation from the last tower summaries.
            self.summary_op = tf.summary.merge(self.summaries)

            self.summary_writer = tf.summary.FileWriter(cfg.MODEL.model_path, self.sess.graph)
            self.summary_writer_val = tf.summary.FileWriter(cfg.MODEL.model_path + '_val', self.sess.graph)

            min_loss_control=1000.
            for epoch in tqdm.tqdm(range(cfg.TRAIN.epoch_start,cfg.TRAIN.epoch)):
                self._train(epoch)
                val_loss=self._val(epoch)
                # logger.info('**************'
                #            'val_loss %f '%(val_loss))
                # self.Logg.summarize(epoch,'test',summaries_dict={'val_loss':val_loss})

                #tmp_model_name=cfg.MODEL.model_path + \
                #               'epoch_' + str(epoch ) + \
                #               'L2_' + str(cfg.TRAIN.weight_decay_factor) + \
                #               '.ckpt'
                #logger.info('save model as %s \n'%tmp_model_name)
                #self.saver.save(self.sess, save_path=tmp_model_name)

                if val_loss < min_loss_control:
                    min_loss_control=val_loss
                    low_loss_model_name = cfg.MODEL.model_path + \
                                     'epoch_' + str(epoch) + \
                                     'L2_' + str(cfg.TRAIN.weight_decay_factor)  + '.ckpt'
                    logger.info('A new low loss model  saved as %s \n' % low_loss_model_name)
                    self.saver.save(self.sess, save_path=low_loss_model_name)

            self.sess.close()



    def _train(self,_epoch):
        running_l2_loss = []
        running_reg_loss = []
        running_cla_loss = []
        running_total_loss = []
        running_lr_value = []
        # running_mAP_05 = []

        for step in tqdm.tqdm(range(cfg.TRAIN.iter_num_per_epoch)):
            self.ite_num += 1

            ########show_flag check the data
            if cfg.TRAIN.vis:
                example_image,example_all_boxes,example_all_labels = next(self.train_ds)

                print(example_all_boxes.shape)
                for i in range(example_all_boxes.shape[0]):


                    img=example_image[i]
                    box_encode=example_all_boxes[i]
                    label=example_all_labels[i]

                    print(np.sum(label[label>0]))
                    # for j in range(label.shape[0]):
                    #     if label[j]>0:
                    #         print(box_encode[j])
                    cv2.namedWindow('img', 0)
                    cv2.imshow('img', img)
                    cv2.waitKey(0)

            else:

                start_time = time.time()
                feed_dict = {}
                examples = next(self.train_ds)
                for n in range(cfg.TRAIN.num_gpu):



                    feed_dict[self.inputs[0][n]] = examples[0][n*cfg.TRAIN.batch_size:(n+1)*cfg.TRAIN.batch_size]
                    feed_dict[self.inputs[1][n]] = examples[1][n*cfg.TRAIN.batch_size:(n+1)*cfg.TRAIN.batch_size]
                    feed_dict[self.inputs[2][n]] = examples[2][n*cfg.TRAIN.batch_size:(n+1)*cfg.TRAIN.batch_size]

                feed_dict[self.inputs[3]] = cfg.TRAIN.weight_decay_factor
                feed_dict[self.inputs[4]] = True

                fetch_duration = time.time() - start_time

                start_time2 = time.time()
                _, total_loss_value,reg_loss_value,cla_loss_value,l2_loss_value,lr_value,result_network = \
                    self.sess.run([*self.outputs],
                             feed_dict=feed_dict)

                # result_network['labels'] in [-1,0] for 1 class
                # result_network['labels'] in [-1,0,1] for 2 class

                # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
                # gt = np.array([
                #     [439, 157, 556, 241, 0, 0, 0],
                # ])
                #
                # # [xmin, ymin, xmax, ymax, class_id, confidence]
                # preds = np.array([
                #     [429, 219, 528, 247, 0, 0.460851],
                # ])

                # boxes = np.reshape(result_network['boxes'],(-1,4))
                # labels = np.reshape(result_network['labels'],(-1,1))
                # scores = np.reshape(result_network['scores'],(-1,1))
                #
                # pred = np.append(boxes,labels,1)
                # pred = np.append(pred, scores, 1)
                # pred = pred[pred[:, 4] == 0]
                #
                # boxes = examples[1][n*cfg.TRAIN.batch_size:(n+1)*cfg.TRAIN.batch_size]
                # labels = examples[2][n*cfg.TRAIN.batch_size:(n+1)*cfg.TRAIN.batch_size]-1
                # boxes = np.reshape(boxes, (-1,4))
                # labels = np.reshape(labels, (-1,1))
                # z = np.zeros_like(labels)
                #
                # gt = np.append(boxes,labels,1)
                # gt = np.append(gt, z, 1)
                # gt = np.append(gt, z, 1)
                # gt = gt[gt[:, 4] == 0]


                # metric_fn.add(pred, gt)
                # mAP = metric_fn.value(iou_thresholds=np.arange(0.5), mpolicy='soft')['mAP']
                # metric_fn.reset()

                running_l2_loss += [l2_loss_value]
                running_reg_loss += [reg_loss_value]
                running_cla_loss += [cla_loss_value]
                running_total_loss += [total_loss_value]
                running_lr_value += [lr_value]
                # running_mAP_05 += [mAP]

        neptune.log_metric('training_l2_loss', np.mean(running_l2_loss))
        neptune.log_metric('training_reg_loss', np.mean(running_reg_loss))
        neptune.log_metric('training_cla_loss', np.mean(running_cla_loss))
        neptune.log_metric('training_total_loss', np.mean(running_total_loss))
        # neptune.log_metric('training_mAP_05', np.mean(running_mAP_05))
        neptune.log_metric('lr', np.mean(running_lr_value))
                # duration = time.time() - start_time2
                # run_duration = duration
                # if self.ite_num % cfg.TRAIN.log_interval == 0:
                #     num_examples_per_step = cfg.TRAIN.batch_size * cfg.TRAIN.num_gpu
                #     examples_per_sec = num_examples_per_step / duration
                #     sec_per_batch = duration / cfg.TRAIN.num_gpu
                #
                #
                #
                #
                #     format_str = ('epoch %d: iter %d, '
                #                   'total_loss=%.6f '
                #                   'reg_loss=%.6f '
                #                   'cla_loss=%.6f '
                #                   'l2_loss=%.6f '
                #                   'learning rate =%e '
                #                   '(%.1f examples/sec; %.3f sec/batch) '
                #                   'fetch data time = %.6f'
                #                   'run time = %.6f')
                #     # logger.info(format_str % (_epoch,
                #     #                           self.ite_num,
                #     #                           total_loss_value,
                #     #                           reg_loss_value,
                #     #                           cla_loss_value,
                #     #                           l2_loss_value,
                #     #                           lr_value,
                #     #                           examples_per_sec,
                #     #                           sec_per_batch,
                #     #                           fetch_duration,
                #     #                           run_duration))
                #
                # if self.ite_num % 100 == 0:
                #     summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                #     self.summary_writer.add_summary(summary_str, self.ite_num)


    def _val(self,_epoch):
        running_l2_loss = []
        running_reg_loss = []
        running_cla_loss = []
        running_total_loss = []
        # running_mAP_05 = []

        all_total_loss=0
        for step in range(cfg.TRAIN.val_iter):

            feed_dict = {}
            examples = next(self.val_ds)
            for n in range(cfg.TRAIN.num_gpu):


                feed_dict[self.inputs[0][n]] = examples[0][n*cfg.TRAIN.batch_size:(n+1)*cfg.TRAIN.batch_size]
                feed_dict[self.inputs[1][n]] = examples[1][n*cfg.TRAIN.batch_size:(n+1)*cfg.TRAIN.batch_size]
                feed_dict[self.inputs[2][n]] = examples[2][n*cfg.TRAIN.batch_size:(n+1)*cfg.TRAIN.batch_size]

            feed_dict[self.inputs[3]] = 0
            feed_dict[self.inputs[4]] = False

            total_loss_value, reg_loss_value,cla_loss_value, l2_loss_value, lr_value, result_network = \
                self.sess.run([*self.val_outputs],
                              feed_dict=feed_dict)

            # boxes = np.reshape(result_network['boxes'], (-1, 4))
            # labels = np.reshape(result_network['labels'], (-1, 1))
            # scores = np.reshape(result_network['scores'], (-1, 1))
            #
            # pred = np.append(boxes, labels, 1)
            # pred = np.append(pred, scores, 1)
            # pred = pred[pred[:, 4] == 0]
            #
            # boxes = examples[1][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
            # labels = examples[2][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size] - 1
            # boxes = np.reshape(boxes, (-1, 4))
            # labels = np.reshape(labels, (-1, 1))
            # z = np.zeros_like(labels)
            #
            # gt = np.append(boxes, labels, 1)
            # gt = np.append(gt, z, 1)
            # gt = np.append(gt, z, 1)
            # gt = gt[gt[:, 4] == 0]
            #
            # metric_fn.add(pred, gt)
            # mAP = metric_fn.value(iou_thresholds=np.arange(0.5), mpolicy='soft')['mAP']
            # metric_fn.reset()

            all_total_loss+=total_loss_value

            running_l2_loss += [l2_loss_value]
            running_reg_loss += [reg_loss_value]
            running_cla_loss += [cla_loss_value]
            running_total_loss += [total_loss_value]
            # running_mAP_05 += [mAP]

        neptune.log_metric('validation_l2_loss', np.mean(running_l2_loss))
        neptune.log_metric('validation_reg_loss', np.mean(running_reg_loss))
        neptune.log_metric('validation_cla_loss', np.mean(running_cla_loss))
        neptune.log_metric('validation_total_loss', np.mean(running_total_loss))
        # neptune.log_metric('validation_mAP_05', np.mean(running_mAP_05))


        return all_total_loss/cfg.TRAIN.val_iter



    def train(self):
        self.train_loop()


    def eva_in_val(self):
        self.build()
        self.load_weight()

        print("start evaluation")

        with self._graph.as_default():
            # Create a saver.
            running_mAP_05 = []
            ds = self.val_metric_gen
            for step in tqdm.tqdm(range(cfg.TRAIN.val_iter)):

                feed_dict = {}
                examples = next(self.val_ds)
                for n in range(cfg.TRAIN.num_gpu):
                    feed_dict[self.inputs[0][n]] = examples[0][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
                    feed_dict[self.inputs[1][n]] = examples[1][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
                    feed_dict[self.inputs[2][n]] = examples[2][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]

                feed_dict[self.inputs[3]] = 0
                feed_dict[self.inputs[4]] = False

                total_loss_value, reg_loss_value, cla_loss_value, l2_loss_value, lr_value, result_network = \
                    self.sess.run([*self.val_outputs],
                                  feed_dict=feed_dict)


                num_boxes = result_network['num_boxes']

                boxes = np.reshape(result_network['boxes'], (-1, 4))
                boxes *= examples[0].shape[1]
                labels = np.reshape(result_network['labels'], (-1, 1))
                scores = np.reshape(result_network['scores'], (-1, 1))

                pred = np.append(boxes, labels, 1)
                pred = np.append(pred, scores, 1)
                pred = pred[pred[:, 4] == 0]

                pred_n = np.zeros_like(pred)
                pred_n[:,0] = pred[:,1]
                pred_n[:,1] = pred[:,0]
                pred_n[:,2] = pred[:,3]
                pred_n[:,3] = pred[:,2]
                pred_n[:,4] = pred[:,4]
                pred_n[:,5] = pred[:,5]

                pred = pred_n

                pred = pred[pred[: , 5] > 0.5]

                examples = ds[step]
                boxes = examples[1]
                labels = examples[2] - 1
                boxes = np.reshape(boxes, (-1, 4))
                labels = np.reshape(labels, (-1, 1))
                z = np.zeros_like(labels)

                gt = np.append(boxes, labels, 1)
                gt = np.append(gt, z, 1)
                gt = np.append(gt, z, 1)
                gt = gt[gt[:, 4] == 0]


                if len(pred) == 0 and len(gt) == 0:
                    mAP = 1
                else:
                    metric_fn.add(pred, gt)
                    mAP = metric_fn.value(iou_thresholds=np.arange(0.5))['mAP']
                    metric_fn.reset()
                running_mAP_05 += [mAP]

            val_map = np.mean(running_mAP_05)

            running_mAP_05 = []
            ds = self.train_metric_gen

            iter_n = cfg.TRAIN.train_set_size// cfg.TRAIN.num_gpu // cfg.TRAIN.batch_size
            for step in tqdm.tqdm(range(iter_n)):

                feed_dict = {}
                examples = next(self.trainds_for_eval)
                for n in range(cfg.TRAIN.num_gpu):
                    feed_dict[self.inputs[0][n]] = examples[0][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
                    feed_dict[self.inputs[1][n]] = examples[1][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
                    feed_dict[self.inputs[2][n]] = examples[2][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]

                feed_dict[self.inputs[3]] = 0
                feed_dict[self.inputs[4]] = False

                total_loss_value, reg_loss_value, cla_loss_value, l2_loss_value, lr_value, result_network = \
                    self.sess.run([*self.val_outputs],
                                  feed_dict=feed_dict)

                num_boxes = result_network['num_boxes']

                boxes = np.reshape(result_network['boxes'], (-1, 4))
                boxes *= examples[0].shape[1]
                labels = np.reshape(result_network['labels'], (-1, 1))
                scores = np.reshape(result_network['scores'], (-1, 1))

                pred = np.append(boxes, labels, 1)
                pred = np.append(pred, scores, 1)
                pred = pred[pred[:, 4] == 0]

                pred_n = np.zeros_like(pred)
                pred_n[:, 0] = pred[:, 1]
                pred_n[:, 1] = pred[:, 0]
                pred_n[:, 2] = pred[:, 3]
                pred_n[:, 3] = pred[:, 2]
                pred_n[:, 4] = pred[:, 4]
                pred_n[:, 5] = pred[:, 5]

                pred = pred_n

                pred = pred[pred[:, 5] > 0.5]

                examples = ds[step]
                boxes = examples[1]
                labels = examples[2] - 1
                boxes = np.reshape(boxes, (-1, 4))
                labels = np.reshape(labels, (-1, 1))
                z = np.zeros_like(labels)

                gt = np.append(boxes, labels, 1)
                gt = np.append(gt, z, 1)
                gt = np.append(gt, z, 1)
                gt = gt[gt[:, 4] == 0]

                if len(pred) == 0 and len(gt) == 0:
                    mAP = 1
                else:
                    metric_fn.add(pred, gt)
                    mAP = metric_fn.value(iou_thresholds=np.arange(0.5))['mAP']
                    metric_fn.reset()
                running_mAP_05 += [mAP]

            train_map = np.mean(running_mAP_05)
            print(cfg.MODEL.model_path)
            print('training metric mAP05: ', train_map)
            print('validation metric mAP05: ', val_map)

            self.sess.close()



    def save_val_result(self):
        self.build()
        self.load_weight()

        print("start evaluation")
        dest_folder = os.path.join('samples_result', cfg.MODEL.model_path)
        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)

        gt_path = os.path.join(dest_folder,'img_gt')
        if not os.path.isdir(gt_path):
            os.mkdir(gt_path)

        pred_path = os.path.join(dest_folder, 'img_pred')
        if not os.path.isdir(pred_path):
            os.mkdir(pred_path)

        with self._graph.as_default():
            # Create a saver.
            running_mAP_05 = []
            ds = self.val_metric_gen
            for step in tqdm.tqdm(range(cfg.TRAIN.val_iter)):

                feed_dict = {}
                examples = next(self.val_ds)
                img_from_gen = examples[0][0]
                for n in range(cfg.TRAIN.num_gpu):
                    feed_dict[self.inputs[0][n]] = examples[0][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
                    feed_dict[self.inputs[1][n]] = examples[1][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
                    feed_dict[self.inputs[2][n]] = examples[2][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]

                feed_dict[self.inputs[3]] = 0
                feed_dict[self.inputs[4]] = False
                # print(np.unique( examples[2][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]))
                # exit()
                total_loss_value, reg_loss_value, cla_loss_value, l2_loss_value, lr_value, result_network = \
                    self.sess.run([*self.val_outputs],
                                  feed_dict=feed_dict)

                img_full = examples[0][0]
                num_boxes = result_network['num_boxes']

                boxes = np.reshape(result_network['boxes'], (-1, 4))
                # print(np.unique(boxes))
                # continue
                boxes *= examples[0].shape[1]
                labels = np.reshape(result_network['labels'], (-1, 1))
                scores = np.reshape(result_network['scores'], (-1, 1))
                # print(np.unique(labels))
                # continue
                # exit()
                pred = np.append(boxes, labels, 1)
                pred = np.append(pred, scores, 1)
                pred = pred[pred[:, 4] == 0]

                # pred = pred[pred[: , 5] > 0.5]

                examples = ds[step]
                boxes = examples[1]
                labels = examples[2] - 1
                boxes = np.reshape(boxes, (-1, 4))
                labels = np.reshape(labels, (-1, 1))
                z = np.zeros_like(labels)

                gt = np.append(boxes, labels, 1)
                gt = np.append(gt, z, 1)
                gt = np.append(gt, z, 1)
                gt = gt[gt[:, 4] == 0]


                img_full_gt = img_full.copy()
                for box in gt:
                    xmin = int(box[0])
                    ymin = int(box[1])
                    xmax = int(box[2])
                    ymax = int(box[3])
                    img_full_gt = cv2.rectangle(img_full_gt, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                img_full_gt = cv2.cvtColor(img_full_gt, cv2.COLOR_BGR2RGB)
                cv2.imwrite(gt_path + f'/{step}.png', img_full_gt)

                img_from_gen_gt = img_from_gen.copy()
                for box in pred:
                    if box[5] > 0.5:
                        xmin = int(box[1])
                        ymin = int(box[0])
                        xmax = int(box[3])
                        ymax = int(box[2])
                        img_from_gen_gt = cv2.rectangle(img_from_gen_gt, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                img_from_gen_gt = cv2.cvtColor(img_from_gen_gt, cv2.COLOR_BGR2RGB)
                cv2.imwrite(pred_path + f'/{step}.png', img_from_gen_gt)

            self.sess.close()


    def freeze(self):
        print('freeze..')
        output_node_names = 'tower_0/images,tower_0/boxes,tower_0/scores,tower_0/labels,tower_0/num_detections'
        output_graph = '/media/newdrive/andrew/DSFD-tensorflow_experiments/model_endlersaning/detector_cast_freeze.pb'

        with self._graph.as_default(), tf.device('/cpu:0'):

            # Create an optimizer that performs gradient descent.
            opt, lr, global_step = self.get_opt()

            ##some global placeholder
            L2_reg = tf.placeholder(tf.float32, name="L2_reg")
            # training = tf.placeholder(tf.bool, name="training_flag")
            training = tf.placeholder_with_default(False, (), name="training_flag")

            total_loss_to_show = 0.
            images_place_holder_list = []
            labels_place_holder_list = []
            boxes_place_holder_list = []

            weights_initializer = slim.xavier_initializer()
            biases_initializer = tf.constant_initializer(0.)
            biases_regularizer = tf.no_regularizer
            weights_regularizer = tf.contrib.layers.l2_regularizer(L2_reg)

            # Calculate the gradients for each model tower.
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_0') as scope:
                    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

                        images_ = tf.placeholder(tf.float32, [None, None, None, 3], name="images")
                        boxes_ = tf.placeholder(tf.float32, [cfg.TRAIN.batch_size, None, 4], name="input_boxes")
                        labels_ = tf.placeholder(tf.int64, [cfg.TRAIN.batch_size, None], name="input_labels")
                        ###total anchor

                        images_place_holder_list.append(images_)
                        labels_place_holder_list.append(labels_)
                        boxes_place_holder_list.append(boxes_)

                        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                                             slim.conv2d_transpose, slim.separable_conv2d,
                                             slim.fully_connected],
                                            weights_regularizer=weights_regularizer,
                                            biases_regularizer=biases_regularizer,
                                            weights_initializer=weights_initializer,
                                            biases_initializer=biases_initializer):
                            reg_loss, cla_loss, l2_loss, result_network = self.tower_loss(
                                scope, images_, labels_, boxes_, L2_reg, training)

            #                 ##use muti gpu ,large batch
            #                 # if i == cfg.TRAIN.num_gpu - 1:
            #                 #     total_loss = tf.add_n([reg_loss, cla_loss, l2_loss])
            #                 # else:
            #                 total_loss = tf.add_n([reg_loss, cla_loss])
            #             total_loss_to_show += total_loss
            #             # Reuse variables for the next tower.
            #             tf.get_variable_scope().reuse_variables()
            #
            #             ##when use batchnorm, updates operations only from the
            #             ## final tower. Ideally, we should grab the updates from all towers
            #             # but these stats accumulate extremely fast so we can ignore the
            #             #  other stats from the other towers without significant detriment.
            #             bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            #
            #             # Retain the summaries from the final tower.
            #             self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            #
            #             ###freeze some params
            #             train_var_list = self.frozen()
            #             # Calculate the gradients for the batch of data on this CIFAR tower.
            #             grads = opt.compute_gradients(total_loss, train_var_list)
            #
            #             # Keep track of the gradients across all towers.
            #             tower_grads.append(grads)
            # # We must calculate the mean of each gradient. Note that this is the
            # # synchronization point across all towers.
            # grads = self.average_gradients(tower_grads)
            #
            # # Add a summary to track the learning rate.
            # self.add_summary(tf.summary.scalar('learning_rate', lr))
            # self.add_summary(tf.summary.scalar('total_loss', total_loss_to_show))
            # self.add_summary(tf.summary.scalar('loc_loss', reg_loss))
            # self.add_summary(tf.summary.scalar('cla_loss', cla_loss))
            # self.add_summary(tf.summary.scalar('l2_loss', l2_loss))
            #
            # # Add histograms for gradients.
            # for grad, var in grads:
            #     if grad is not None:
            #         self.add_summary(tf.summary.histogram(var.op.name + '/gradients', grad))
            #
            # # Apply the gradients to adjust the shared variables.
            # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
            #
            # # Add histograms for trainable variables.
            # for var in tf.trainable_variables():
            #     self.add_summary(tf.summary.histogram(var.op.name, var))
            #
            # if self.ema_weights:
            #     # Track the moving averages of all trainable variables.
            #     variable_averages = tf.train.ExponentialMovingAverage(
            #         0.9, global_step)
            #     variables_averages_op = variable_averages.apply(tf.trainable_variables())
            #     # Group all updates to into a single train op.
            #     train_op = tf.group(apply_gradient_op, variables_averages_op, *bn_update_ops)
            # else:
            #     train_op = tf.group(apply_gradient_op, *bn_update_ops)
            #
            # ###set inputs and ouputs
            # self.inputs = [images_place_holder_list,
            #                boxes_place_holder_list,
            #                labels_place_holder_list,
            #                L2_reg,
            #                training]
            # self.outputs = [train_op,
            #                 total_loss_to_show,
            #                 reg_loss,
            #                 cla_loss,
            #                 l2_loss,
            #                 lr]
            # self.val_outputs = [total_loss_to_show,
            #                     reg_loss,
            #                     cla_loss,
            #                     l2_loss,
            #                     lr]

            # tf_config = tf.ConfigProto(
            #     allow_soft_placement=True,
            #     log_device_placement=False)
            # tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session()

            ##init all variables
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.Logg = Logger(self.sess)
        self.load_weight()
        print('loaded')

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,  # The session is used to retrieve the weights
            self._graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(","),
        )
        # print(type(output_graph_def))
        # print(dir(output_graph_def))
        # tf.contrib.quantize.create_eval_graph(output_graph_def)
        # print(type(output_graph_def))
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    def resave(self):
        self.build()
        self.load_weight()

        with self._graph.as_default():
            # Create a saver.
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            self.saver.save(self.sess, save_path='/media/newdrive/andrew/DSFD-tensorflow_experiments/model_endlersaning/new_resave_epoch_40L2_0.0005.ckpt')



