# chainer-FMD
Image classifier for Flickr Material Database (FMD) in chainer

## Setup

1. GitHubからchainer-FMDをクローンします  
(clone chainer-FMD from GitHub)
```
git clone https://github.com/isl-kobayan/chainer-FMD.git
```
2. [Flickr Material Database](http://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip) をダウンロードします。  
(download [Flickr Material Database](http://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip))
3. "chainer-FMD" フォルダー内で "FMD"フォルダを作成します。   
(create "FMD" directory in "chainer-FMD")
4. FMD.zipを解凍し、"FMD"フォルダー内に"image"フォルダーをコピーします。   
(unzip "FMD.zip" and move "image" directory to "FMD" directory)
5. "scale_images.sh" を実行します。imagemagickが必要なので、インストールされていない場合は以下のコマンドを実行してください。
(execute "scale_images.sh". This needs imagemagick. If it is not installed, execute below command.)
```
sudo apt-get install imagemagick
```
6. 

## モデルをダウンロード
* NIN https://gist.github.com/mavenlin/d802a5849de39225bcc6
* VGG https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
* GoogLeNet https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
* illustration2vec http://illustration2vec.net/   (pre-trained model for tag prediction, version 2.0)
* alexnet https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
