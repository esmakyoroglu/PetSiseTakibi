
#komut satırlarını kolaylıkla yazmamızı sağlar
import argparse 
#framelerin hız oranlarını kullanmak için yaptık
import time 
from pathlib import Path
#opencv kütüphanesi obje tanımlama ve izlemek için sağlamak için
import cv2
# yolov7 pytorch oluşturarak kullanıldığından  performans açısından objeyı tanımlama ve izleme için dahil ederiz
import torch
#pytorch un arkaplanda desteklenebilmesi için davranışlarının kontrolunu sağlayabilen kütüphane 
import torch.backends.cudnn as cudnn
#random sayı üretibelen fonksiyonu dahil ederiz her bir şise için daha sonra ıd ataması yapıcaz random sayı ile
from numpy import random
#deque, daha hızlı ekkleme ve  pop işlemleri yapabilmek için liste yerine tercih edilir
from collections import deque
#listeyi diziye çevirmemizde yardımcı olacak
import numpy as np
#uzaklık hesaplamalarını yapalbilmek için ekledik
import math
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from deep_sort_pytorch.utils.parser import get_config
#objeyi izlemeyi deepsort sayesinde sağlamış olucaz
from deep_sort_pytorch.deep_sort import DeepSort


#birden fazla renk tanımlanıştır farklı obje tespiti için farklı bir id sayesinde yeni bir renk oluşturmamızı sağlar
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1) #renk paletini tanımlama 
data_deque = {} # data_deque adıyla bir Sözlüğü başlatma, tanımlanan nesnlerin isimleri için sözlük oluşturuyoruz
obje_sayac = {} #algılanan nesne içi nesne sözlüğü oluşturma, ekranda tespit edilen nesnleri yazdırabilmek için, nesne tespitini gösterebilmek için oluşturmamız gereken sözlük
obje_id= {} #nesne izlemeyi kullanarak her birine benzersiz bir id atayacağız
line = [(348,0), (348, 356)] #videoda manuel olarak çizgi oluşturuyoruz tam ortasına. nesne, oluşturulan çizgiden geçtiğinde nesne ile çigi kesiştiğinde sayacı artırımamızı sağlayacak

#fonksiyonda yolov7 den aldığımız hazır nesnenin çıktılarını alıyoruz
#deepSort algoritmasıyla formatıyla uyumlu bir şekilde
#bu fonksiyon x ve y kordinatlarını merkez  x_c ve y_c olmak üzere kutucuğun 
#yükseliğinin ve genişliğine döndürür
#sınırlayıcı kutuyu mutlak piksel değerlerinden hesaplar
def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    #merkezini alıyoruz
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

#bu fonksiyonda nesneyi takip edebilmemizi sağlayan kutucuğun üst ve alt tarafın genişlik ve yükselik kordinatlarına odaklanırız 
def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

#etikerlerimiz için renkleri belirliyoruz
def renk_paleti(label):
    if label == 0: #insan
        renk = (85,45,255)
    elif label == 2: # araba
        renk = (222,82,175)
    elif label == 3:  # Motosiklet
        renk = (0, 204, 255)
    elif label == 5:  # otobüs
        renk = (0, 149, 255)
    else:
        renk = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(renk)
# yuvarlatılmış dikdörtgen köşelerini olusturuyoruz ardından bu fonksiyon döndürelen parametrelerini metin dosyasında saklayıp tekrar kullanıcaz
def sinirlari_ciz(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # sol üst taraf
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # sağ üst taraf
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # sol alt
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # sağ alt
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img
#l: köşe kenarının uzunluğu, t köşe kenarlarının kalınlığı, rt dikdörtgenin kalınlığını belirleme
#colorR: dikdörtgenin sahip olacağı renk, colorC:köşelerin rengleri 
def dikdortgen_kose(img, bbox, l=30, t=5, rt=1,
               colorR=(255, 0, 255), colorC=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if rt != 0:
        cv2.rectangle(img, bbox, colorR, rt)
    # sol üst x,y kordinatları çizgileri
    cv2.line(img, (x, y), (x + l, y), colorC, t)
    cv2.line(img, (x, y), (x, y + l), colorC, t)
    # sağ üst tarf  x1,y kordinatları çizgileri
    cv2.line(img, (x1, y), (x1 - l, y), colorC, t)
    cv2.line(img, (x1, y), (x1, y + l), colorC, t)
    # sol alt  x,y1
    cv2.line(img, (x, y1), (x + l, y1), colorC, t)
    cv2.line(img, (x, y1), (x, y1 - l), colorC, t)
    # sağ alt  x1,y1
    cv2.line(img, (x1, y1), (x1 - l, y1), colorC, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), colorC, t)

    return 
# Görüntü üzerinde sınırlayıcı kutu cv2.rectangle opencv ile gelen bir özellik sayesinde tespit edilen nesne üzerinden geçer
# çizilen yuvarlatılmış dikdörtgen içerisinde hangi nesne varsa cv2.text içerisinde yazan nesne isimlerine göre etiket ekliyoruz  
# daha önceden oluşturduğumuz sinirlari_ciz fonksiyonu sayesinde oluşturulan etiket eklenir
def UI_box(x, img, color=None, label=None, line_thickness=None):
    #her bir video framelerini tek tek şişelere kutucuk çiziyoruz
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1 
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    x1=int(x[0])
    y1=int(x[1])
    x2=int(x[2])
    y2=int(x[3])
    w,h=x2-x1,y2-y1
# her bir etikete göre ayrı bounding box oluşturan fonksiyon, oluşan nesne üzerinde isimleride yazıcak
# cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA) alınacak parametreler
    dikdortgen_kose(img, (x1,y1,w,h), l=9, rt=2,colorR=(255,0,255))
    if label:
        #yazı kalınlığı
        tf = max(tl - 1, 1)  
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = sinirlari_ciz(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

#eğer nesne oluşturduğumuz yeşil merkez çizgiden geçerse şise sayısının arttırıcaz bunun merkeze koyduğumuz nokta sayesinde sağlayabiliyoruz
def intersect(A, B, C,D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
#her bir obje için benzersiz id ataması
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0]-A[0])
# artık alınan her bir değerin resim üzerinde kare kare algılama yaptığımız için mevcut karenin yüksekliğini ve genişilğini çiziyoruz
def kutulari_ciz(img, bbox, names, obje_id, identities=None, offset=(0, 0)):
    #her bir frame için yükselik ve genişlik kontrolü yapılır
    cv2.line(img, line[0], line[1], (46,162,112), 3)
    height, width, _ = img.shape 
    #algılanan nesnenin kimliğini nesne çerçeveye girdiği an saklayacağız
    #nesne kaybolursa izlenen nokta arabellekten kaldırılır id si silinir data_deque den 
    #yeni bir obje tanıdığında ona atanıcak olan idi yi  kaydeder, kaybederse siler
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)#nesne artık çerçevede değilse data_deque listesinden nesnenin kimliğini kaldırırız
    #burada sınırlayıcı kutular arasında  sırasıyla birer birer dolaşacağız sahip olduğumuz kordinatları
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
    # oluşan her bir kutucuğun merkezini bulur
    center = (int((x2+x1)/ 2), int((y1+y2)/2)) #sınırlayıcı kutunun ortası
    cv2.circle(img, center, 2, (0,255,0), -1)
    # her nesnenin benzersiz kimliğini alır etiketler sayesinde
    id = int(identities[i]) if identities is not None else 0
    #eğer nesne framede görüntülenirse ona benzersiz id atama işlemini yapmıştık
    #bundan sonra yeni bir arabellek oluşturmalıyız maksimum uzunluğu 64 olan eğer 65. elementi eklemeye çalışırsak ilk element silinecek arabellekten
    #her yeni bir buffer yeni bir obje için oluşturulmuş olur
    if id not in data_deque:  
          data_deque[id] = deque(maxlen= opt.trailslen)
        #tanımlanan her bir nesne için benzersiz renk ataması yapılır yuvarlatılmış dikdörtgenler için
    #nesne id sayesinde daha önceden etiklediğimiz nesne adını buluruz.
    obje_name = names[obje_id[i]]
    #etiketin gerekli formatta ayarlanması
    label = '{}{:d}'.format("", id) + ":"+ '%s' % (obje_name)
    
    #nesneyi izleyen kutucuğun her seferinde merkez konumunu kullanarak nesneyi farklı bir framede gördüğümüzde
    #bir çizgi oluşturmak için merkezleri birleştiricez bu şekilde nesneyi izleme yolunu anlamış olucaz
    data_deque[id].appendleft(center)
    
    if len(data_deque[id]) >= 2:
#orta çizgiden geçtiği an bir şise idsini çekip, şise sayısını artıyoruz
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                if obje_name not in obje_sayac:
                    obje_sayac[obje_name] = 1
                else:
                    obje_sayac[obje_name] += 1

    UI_box(box, img, label=label, color=color, line_thickness=2)
    
    for idx, (key, value) in enumerate(obje_sayac.items()):
        cnt_str = "Bottles" + ": " + str(value)
        cv2.line(img, (width - 175 ,25+ (idx*40)), (width,25 + (idx*40)), [85,45,255], 30)
        cv2.putText(img, cnt_str, (width - 175, 35 + (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    return img

def yukle(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names)) 
def detect(save_img=False):
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace 
    save_img = not opt.nosave and not source.endswith('.txt') 
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    # deepsort algoritması başlangıcı
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'
    #modelin yüklenmesi 
    model = attempt_load(weights, map_location=device)  #model FP32 yükleme
    stride = int(model.stride.max()) 
    imgsz = check_img_size(imgsz, s=stride)  
    
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16
    
    #ikinci aşama sınıflandırıcı
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    #veri yükleyici
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

  #isimlerin ve renklerin alınması
    names = yukle(names)
   
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) 
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]
        #sonucun çıkarılması
        t1 = time_synchronized()
        with torch.no_grad():   
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):  #her bir görüntü için algılama
            if webcam:  
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p) 
            save_path = str(save_dir / p.name) # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
            if len(det):
                # Kutuları img_size'den im0 boyutuna yeniden ölçeklendirme
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                #sonucları görüntülüleme yazdırma
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  #her bir sınıf için algılama
                    s += '%g %ss, ' % (n, names[int(c)])  #siziye ekleme
                
                xywh_bboxs = [] # Sınırlayıcı kutuların genişliği ve yüksekliği ile merkez koordinatlarımızı içerecek olan başlatma listesi
                #yukarıdakiyle aynı bu listeye güven değeri ekleyeceğiz
                confs = []
                #nesne kimliklerini ekleyeceğiz
                oids = []
                # sonuçları yazdırma tespit ettiğimiz tüm değerlerin üzerinden geçiyoruz
                for *xyxy, conf, cls in reversed(det):
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))
                    if save_txt:  #dosyaya yazdırma
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                #burada derin sıralama güncelleme fonksiyonunu kullanıyoruz ve deepsort id ataması yapıyoruz
                ciktilar = deepsort.update(xywhs, confss, oids, im0)
                if len(ciktilar) > 0:
                    bbox_xyxy = ciktilar[:, :4]
                    identities = ciktilar[:, -2]
                    object_id = ciktilar[:, -1]
                    # Sınırlayıcı kutuları çizmek, etiketlemek ve izleyicinin kimliğini göstermek için DrawBoxes işlevi
                    kutulari_ciz(im0, bbox_xyxy, names, object_id,identities)
            #yazdirma süresi
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
                
            # Sonuçları kaydet 
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else: 
                    if vid_path != save_path:  #yeni video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else: 
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
      

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  #tüm modelleri güncelleme hatayı düzeltmek için
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
