import os
from recnet.picture_preprocessor.resize import splitimage

if __name__ == '__main__':

    src = '/home/cheung/Desktop/redata/0_Label_1.jpg'
    dstpath = '/home/cheung/Desktop'
    if os.path.isfile(src):

        if (dstpath == '') or os.path.exists(dstpath):
            row = 1
            col = 4
            if row > 0 and col > 0:
                splitimage(src, row, col, dstpath)
            else:
                print('无效的行列切割参数！')
        else:
            print('图片输出目录 %s 不存在！' % dstpath)
    else:
        print('图片文件 %s 不存在！' % src)