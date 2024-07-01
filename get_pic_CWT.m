%% 获取连续小波变换时频二维数据
function cwt_rgb = get_pic_CWT(seismicdata, motherwavelet)
% 使用CWT得到时频图
[cfs,~] = cwt(seismicdata, motherwavelet);
% VGG19需要的图像尺寸
imgSize = [224, 224]; 
% mat2gray  用于将数据映射到 [0, 1] 的范围
% im2uint8  用于将数据转换为 8 位无符号整数
% ind2rgb   用于将灰度图映射为 RGB 图像。
% jet(m)    返回具有m种颜色的颜色图
im = ind2rgb(im2uint8(mat2gray(abs(cfs))), jet(256));
cwt_rgb = imresize(im, imgSize);
end