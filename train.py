import torch
import resnet
import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from tensorboardX import SummaryWriter

import os
import argparse

import torch.nn as nn
import torch.nn.init as init

import trimesh
import raster

scene_tri_file = '3dmodels/triangle.obj'
scene_tri = trimesh.load(scene_tri_file)
scene_rec_file = '3dmodels/reectangle.obj'
scene_rec = trimesh.load(scene_rec_file)
# scene.show()

img_w = 100
img_h = 100


IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_DEV = (0.2023, 0.1994, 0.2010)

def unnormalize(img):
    # img: b, ch, h, w
    mean = torch.tensor(IMG_MEAN)[None,:,None,None].to(img.device)
    std  = torch.tensor(IMG_DEV )[None,:,None,None].to(img.device)
    return (img*std)+mean

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)

class ResnetTrainer:

    def __init__(self,
                 resnet_input=None,
                 loss_fn=None,
                 optimizer=None,
                 num_epochs=100,
                 dataset=None,
                 validation_dataset=None,
                 eval_frequency=4,
                 log_dir=None,
                 model_dir='models/',
                 resume=False,
                 learning_rate=0.1,
                 init_params_flag=False,
                 **model_kwargs):

        self.best_acc = 0
        self.start_epoch = 0

        print('Models  at', model_dir)
        self.model_dir = model_dir
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = resnet_input
        if self.model is None:
            # self.model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], **model_kwargs)
            self.model = resnet.ResNetPrunedDepth(resnet.BasicBlock, [2, 2, 2, 2], **model_kwargs)
            # self.model = resnet.ResNetPrunedWidth(resnet.BasicBlock, [2, 2, 2, 2], **model_kwargs)
            # self.model = resnet.ResNetPrunedWidthDepth(resnet.BasicBlock, [2, 2, 2, 2], **model_kwargs)
            if init_params_flag:
                init_params(self.model)

        if resume:
            print('==> Resuming from checkpoint..')
            assert os.path.isdir(self.model_dir), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(self.model_dir+'/best.t7')
            self.model.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']+1

        self.model.to(self.device)

        self.loss_fn = loss_fn
        if self.loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.loss_fn_noavg = torch.nn.CrossEntropyLoss(reduce=False)

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
             lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 50)

        self.num_epochs = num_epochs
        self.eval_frequency = eval_frequency
        self.dataset = dataset
        self.validation_dataset = validation_dataset

        self.log_dir = log_dir
        if log_dir is not None:
            print('Logging at', log_dir)
            self.logger = SummaryWriter(log_dir)
        else:
            self.logger = None


    def checkpoint(self, acc, epoch):
        print('Saving..')
        state = {
            'net': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(state, self.model_dir+'/'+str(epoch)+'.t7')
        if acc > self.best_acc:
            torch.save(state, self.model_dir+'/best.t7')
            self.best_acc = acc

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs+self.start_epoch):
            self.lr_scheduler.step()
            train_loss = self._train_one_epoch(epoch)
            self.visualize_network(epoch+1)


            test_loss, test_accuracy = self._eval_one_epoch(epoch)
            self.checkpoint(test_accuracy, epoch)

            if self.logger is not None:
                self.logger.add_scalar('loss_test', test_loss, epoch)
                self.logger.add_scalar('accuracy_test', test_accuracy, epoch)
                self.logger.add_scalar('loss_train', train_loss, epoch)

    def _train_one_epoch(self, epoch):
        total = 0.0
        total_loss = 0.0
        for i, data in enumerate(self.dataset, 0):
            inputs, labels, category = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if epoch == 0 and i == 0 and self.logger is not None:
                self.logger.add_graph(self.model, inputs)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            total_loss += loss.item()*inputs.size(0)
            total += inputs.size(0)

            if i % 100 == 0 and i>0:
                print('[%d, %5d] loss: %.8f' %
                      (epoch+1, i+1, total_loss/total))

        print('[%d, %5d] loss: %.6f' %
              (epoch+1, i+1, total_loss/total))

        return total_loss/total

    def _eval_one_epoch(self, epoch):
        correct = 0
        total = 0
        total_loss = 0
        all_features = []
        all_category = []
        all_images = []
        with torch.no_grad():
            for data in self.validation_dataset:
                images, labels, category = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                features, outputs = self.model(images, penultimate=True)
                loss = self.loss_fn(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()*labels.size(0)

                all_features.append(features)
                all_category.append(category)
                all_images.append(images)

        features = torch.cat(all_features, 0)
        category = torch.cat(all_category, 0)
        images = torch.cat(all_images, 0)
        if self.logger is not None:
            self.logger.add_embedding(features, metadata=category.numpy(), label_img=unnormalize(images.cpu()),global_step=epoch, tag='test/embedding')
        avg_loss = total_loss/total
        avg_correct = correct/total
        print('Test loss: %.6f accuracy: %d %%' % (avg_loss, 100 * avg_correct))
        return avg_loss, avg_correct

    def save_eval_features_losses(self):
        all_features = []
        all_labels = []
        all_category = []
        all_images = []
        all_losses = []
        all_prediction = []
        with torch.no_grad():
            for data in self.validation_dataset:
                images, labels, category = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                features, outputs = self.model(images, penultimate=True)
                loss = self.loss_fn_noavg(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                all_features.append(features)
                all_labels.append(labels)
                all_category.append(category)
                all_images.append(images)
                all_losses.append(loss)
                all_prediction.append(predicted)

        features = torch.cat(all_features, 0).cpu()
        labels = torch.cat(all_labels, 0).cpu()
        category = torch.cat(all_category, 0).cpu()
        images = unnormalize(torch.cat(all_images, 0).cpu())
        losses = torch.cat(all_losses, 0).cpu()
        prediction = torch.cat(all_prediction, 0).cpu()

        fname = self.model_dir + '/data_final_test.t7'
        dd = {
            'features':features,
            'labels':labels,
            'category':category,
            'images':images,
            'losses':losses,
            'prediction':prediction
        }
        torch.save(dd, fname)

    def save_train_features_losses(self):
        all_features = []
        all_labels = []
        all_category = []
        all_images = []
        all_losses = []
        all_prediction = []
        for i, data in enumerate(self.dataset, 0):
            images, labels, category = data
            images = images.to(self.device)
            labels = labels.to(self.device)
            features, outputs = self.model(images, penultimate=True)
            loss = self.loss_fn_noavg(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            all_features.append(features.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_category.append(category.detach().cpu())
            all_images.append(images.detach().cpu())
            all_losses.append(loss.detach().cpu())
            all_prediction.append(predicted.detach().cpu())

        features = torch.cat(all_features, 0).cpu()
        labels = torch.cat(all_labels, 0).cpu()
        category = torch.cat(all_category, 0).cpu()
        images = unnormalize(torch.cat(all_images, 0).cpu())
        losses = torch.cat(all_losses, 0).cpu()
        prediction = torch.cat(all_prediction, 0).cpu()

        fname = self.model_dir + '/data_final_train.t7'
        dd = {
            'features':features,
            'labels':labels,
            'category':category,
            'images':images,
            'losses':losses,
            'prediction':prediction
        }
        torch.save(dd, fname)

    def visualize_network(self, epoch):
        if self.logger is None:
            return

        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                module_name = str(module)
                if module.in_channels == 3:
                    tag = module_name
                    weights = module.state_dict()['weight']
                    weights_max = weights.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
                    weights_min = weights.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
                    weights = (weights-weights_min)/(weights_max-weights_min)
                    self.logger.add_images(tag, weights, epoch)
                else:
                    for i in range(module.out_channels):
                        tag = module_name+'/'+str(i)
                        weights = module.state_dict()['weight'][i,:,None,:,:].repeat(1,3,1,1)
                        weights_max = weights.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
                        weights_min = weights.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
                        weights = (weights-weights_min)/(weights_max-weights_min)
                        self.logger.add_images(tag, weights, epoch)

    def visualize_cam(self):
        with torch.no_grad():
            i=0
            for data in self.validation_dataset:
                images, labels, category = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                cam_outputs = self.model.output_cam(images) # b, n, h, w
                for batch in range(cam_outputs.size(0)):
                    tag = 'CAM_'+str(i)+'_'+str(labels[batch].item())
                    i=i+1
                    min_val = cam_outputs[batch, :, :, :].min()
                    max_val = cam_outputs[batch, :, :, :].max()
                    cam = (cam_outputs[batch, :, None, :, :].repeat(1,3,1,1)-min_val)/(max_val-min_val)
                    self.logger.add_images(tag+'/RGB', unnormalize(images[batch,:,:,:].cpu()), 12345)
                    self.logger.add_images(tag+'/CAM', cam, 12345)
                break


class render_tri_rect_dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, download=True, transform=None):
        super().__init__()
        self.transform = transform
        self.train = train
        if train:
            self.size = 10000
        else:
            self.size = 1000

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        theta = (np.random.rand(3)-0.5)*2*math.pi/6
        trans = (np.random.rand(3)-0.5)*2*0.5
        R = utils.eulerAnglesToRotationMatrix(theta)
        t = trans + [0,0,3]
        if idx < self.size/2:
            # Triangle
            image = raster.render_scene(scene_tri.vertices, scene_tri.faces, img_h, img_w, R, t)
            category = 0
        else:
            # Rectangle
            image = raster.render_scene(scene_rec.vertices, scene_rec.faces, img_h, img_w, R, t)
            category = 1
        image = np.repeat(image[None,:,:], 3, axis=0)
        image = torch.from_numpy(image, dtype=torch.float)
        return (img, category)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cifar10 Classification')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume')
    parser.add_argument('--init_params', action='store_true', help='initialize parameters')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=10, type=int, help='num epochs')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader threads')
    parser.add_argument('--log_dir', default=None, type=str, help='log directory')
    parser.add_argument('--model_dir', default='models/classification', type=str, help='model directory')
    args = parser.parse_args()

    num_classes = 10
    eval_frequency = 1

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_DEV)])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_DEV)])

    train_dataset = cifar10_dataset(train=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_workers)
    test_dataset = cifar10_dataset(train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)

    print('Training set of size', len(train_dataset))
    print('Testing  set of size', len(test_dataset))

    trainer = ResnetTrainer(
                    dataset=train_loader,
                    validation_dataset=test_loader,
                    num_classes=num_classes,
                    log_dir=args.log_dir,
                    model_dir=args.model_dir,
                    eval_frequency=eval_frequency,
                    learning_rate=args.lr,
                    num_epochs=args.num_epochs,
                    resume=args.resume,
                    init_params_flag=args.init_params
                )

    trainer.train()
