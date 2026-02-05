"""
For GroundCUA evaluation:

PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --image-dir /proj/docling-vision/users/said/data/yolo_filtered/images/test \
  --labels-dir /proj/docling-vision/users/said/data/yolo_filtered/labels/test \
  --groundcua-root /proj/docling-vision/users/said/GroundCUA/GroundCUA \
  --classes /proj/docling-vision/users/said/data/yolo_filtered/classes.txt \
  --max-samples 870 \
  --screenvlm /proj/docling-vision/users/said/nanoVLM/granite_docling_resume/nanoVLM_siglip2-base-patch16-512_2048_mp4_granite_docling_resume_16xGPU_full_ds_bs64_187500_lr_vision_0.002-language_0.002-0.0212_1209-023045_lsf283674/converted/step_46000 \
  --yolo-model /proj/docling-vision/users/said/webshot-dataset/runs/detect/webshot_ui_refined_labels_filtered/weights/best.pt \
  --qwen3-vl Qwen/Qwen3-VL-8B-Instruct \
  --omniparser-weights /proj/docling-vision/users/said/webshot-dataset/runs/omniparser/model.pt \
  --gemini gemini-2.5-flash-lite \
  --class-schema groundcua \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall,recall_agnostic,ned \
  --batch-size 32
  
  
For YOLO evaluation:

PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --format yolo \
  --image-dir /proj/docling-vision/users/said/data/yolo_filtered/images/test \
  --labels-dir /proj/docling-vision/users/said/data/yolo_filtered/labels/test \
  --classes /proj/docling-vision/users/said/data/yolo_filtered/classes.txt \
  --max-samples 1000 \
  --screenvlm /proj/docling-vision/users/said/nanoVLM/granite_docling_resume/nanoVLM_siglip2-base-patch16-512_2048_mp4_granite_docling_resume_16xGPU_full_ds_bs64_187500_lr_vision_0.002-language_0.002-0.0212_1209-023045_lsf283674/converted/step_46000 \
  --yolo-model /proj/docling-vision/users/said/webshot-dataset/runs/detect/webshot_ui_refined_labels_filtered/weights/best.pt \
  --qwen3-vl Qwen/Qwen3-VL-8B-Instruct \
  --omniparser-weights /proj/docling-vision/users/said/webshot-dataset/runs/omniparser/model.pt \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall,recall_agnostic,ned \
  --batch-size 64
  
  
For ScreenSpot evaluation:
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --image-dir /proj/docling-vision/users/said/data/yolo_filtered/images/test \
  --labels-dir /proj/docling-vision/users/said/data/yolo_filtered/labels/test \
  --screenspot-root /proj/docling-vision/users/said/SeeClick/ScreenSpot \
  --screenspot-split all \
  --classes /proj/docling-vision/users/said/data/yolo_filtered/classes.txt \
  --screenvlm /proj/docling-vision/users/said/nanoVLM/granite_docling_resume/nanoVLM_siglip2-base-patch16-512_2048_mp4_granite_docling_resume_16xGPU_full_ds_bs64_187500_lr_vision_0.002-language_0.002-0.0212_1209-023045_lsf283674/converted/step_46000 \
  --yolo-model /proj/docling-vision/users/said/webshot-dataset/runs/detect/webshot_ui_refined_labels_filtered/weights/best.pt \
  --qwen3-vl Qwen/Qwen3-VL-8B-Instruct \
  --omniparser-weights /proj/docling-vision/users/said/webshot-dataset/runs/omniparser/model.pt \
  --gemini gemini-2.5-flash-lite \
  --class-schema screenspot \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall,recall_agnostic \
  --batch-size 64

For ScreenSpot evaluation with Qwen3-VL:
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --image-dir /proj/docling-vision/users/said/data/yolo_filtered/images/test \
  --labels-dir /proj/docling-vision/users/said/data/yolo_filtered/labels/test \
  --screenspot-root /proj/docling-vision/users/said/SeeClick/ScreenSpot \
  --screenspot-split pc \
  --classes /proj/docling-vision/users/said/data/yolo_filtered/classes.txt \
  --qwen3-vl Qwen/Qwen3-VL-8B-Instruct \
  --class-schema screenspot \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall,recall_agnostic \
  --batch-size 64
  
  
Multi-dataset evaluation (ScreenSpot-web + ScreenSpot-pc + ScreenSpot-mobile + GroundCUA + YOLO@100 + YOLO@1000):
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
  --screenvlm /proj/docling-vision/users/said/nanoVLM/GraniteDoclingV1-stage1-init/nanoVLM_siglip2-base-patch16-512_2048_mp4_GraniteDoclingV1-stage1-init_16xGPU_full_ds_bs64_287500_lr_vision_0.002-language_0.002-0.0212_0109-164905_lsf361208/converted_untied/step_150000 \
  --yolo-model /proj/docling-vision/users/said/webshot-dataset/runs/detect/webshot_ui_refined_labels_filtered/weights/best.pt \
  --qwen3-vl Qwen/Qwen3-VL-8B-Instruct \
  --omniparser-weights /proj/docling-vision/users/said/webshot-dataset/runs/omniparser/model.pt \
  --gemini gemini-2.5-flash-lite \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
  --batch-size 64
  
For ScreenVLM multi-dataset evaluation:  
  
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
  --screenvlm /proj/docling-vision/users/said/nanoVLM/GraniteDoclingV1-stage1-init/nanoVLM_siglip2-base-patch16-512_2048_mp4_GraniteDoclingV1-stage1-init_16xGPU_full_ds_bs64_287500_lr_vision_0.002-language_0.002-0.0212_0109-164905_lsf361208/converted_untied/step_150000 \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
  --batch-size 64
  

For Finetuned Qwen3-VL-2B-Instruct multi-dataset evaluation:

PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
  --screenvlm /proj/docling-vision/users/said/Qwen3-VL/qwen-vl-finetune/checkpoints/Qwen3-VL-2B-Instruct_8GPU_Full_272335/checkpoint-80000 \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
  --batch-size 64
  

For Qwen3-VL-2B-Instruct multi-dataset evaluation:
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
  --qwen3-vl Qwen/Qwen3-VL-2B-Instruct \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
  --batch-size 64

For InternVL3-2B multi-dataset evaluation:
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
  --internvl3 OpenGVLab/InternVL3-2B \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
  --batch-size 64
  
  
For Finetuned Omniparser multi-dataset evaluation (path /proj/docling-vision/users/said/webshot-dataset/runs/detect/omniparser_55c_ft/weights/best.pt):
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
    --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
    --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
    --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
    --yolo-model /proj/docling-vision/users/said/webshot-dataset/runs/detect/omniparser_55c_ft/weights/omniparser_ft.pt \
    --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
    --batch-size 64

For Finetuned Omniparser multi-dataset evaluation (path /proj/docling-vision/users/said/webshot-dataset/runs/detect/omniparser_55c_ft/weights/best.pt):
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=custom55 \
    --yolo-model /proj/docling-vision/users/said/webshot-dataset/runs/detect/omniparser_55c_ft/weights/omniparser_ft_6.pt \
    --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
    --batch-size 64


For Finetuned InternVL3-2B multi-dataset evaluation (path /proj/docling-vision/users/said/InternVL/finetuned/ft_InternVL3-2B/checkpoint-9200):
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
    --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
    --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
    --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
    --screenvlm /proj/docling-vision/users/said/InternVL/finetuned/ft_InternVL3-2B/checkpoint-9200 \
    --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
    --batch-size 64


For RT-DETR multi-dataset evaluation (requires TorchScript exported model):
NOTE: First export your RT-DETR model to TorchScript format (one-time step):
    cd /proj/docling-vision/users/said/RTDETRv2
    .venv/bin/python /tmp/export_torchscript.py \
        -c ./src/rtdetrv2/configs/model/rtdetrv2_r50vd_6x_coco.yml \
        -m ./outputs_rtdetrv2_r50vd_6x_coco/best.pth \
        -o ./outputs_rtdetrv2_r50vd_6x_coco/model.torchscript.pt

Then run evaluation:
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
    --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
    --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
    --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
    --rtdetr /proj/docling-vision/users/said/RTDETRv2/outputs_rtdetrv2_r50vd_6x_coco/model.torchscript.pt \
    --rtdetr-classes /proj/docling-vision/users/said/data/yolo_filtered/classes.txt \
    --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
    --batch-size 64
    
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
    --rtdetr /proj/docling-vision/users/said/RTDETRv2/outputs_rtdetrv2_r50vd_6x_coco/model_gpu.torchscript.pt \
    --rtdetr-classes /proj/docling-vision/users/said/data/yolo_filtered/classes.txt \
    --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
    --batch-size 64
    
    
    
/proj/docling-vision/users/said/nanoVLM/GraniteDoclingV1-stage1-init/nanoVLM_siglip2-base-patch16-512_2048_mp4_GraniteDoclingV1-stage1-init_16xGPU_full_ds_bs64_287500_lr_vision_0.002-language_0.002-0.0212_0121-140230_lsf394476/converted_untied/step_93000
    For ScreenVLM multi-dataset evaluation:  
  
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
  --screenvlm /proj/docling-vision/users/said/nanoVLM/GraniteDoclingV1-stage1-init/nanoVLM_siglip2-base-patch16-512_2048_mp4_GraniteDoclingV1-stage1-init_16xGPU_full_ds_bs64_287500_lr_vision_0.002-language_0.002-0.0212_0121-140230_lsf394476/converted_untied/step_93000 \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
  --batch-size 64
  


/proj/docling-vision/users/said/nanoVLM/GraniteDoclingV1-stage1-init/nanoVLM_siglip2-base-patch16-512_2048_mp4_GraniteDoclingV1-stage1-init_16xGPU_full_ds_bs64_287500_lr_vision_0.002-language_0.002-0.0212_0121-140230_lsf394476/converted_untied/step_107000  
  
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
  --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
  --screenvlm /proj/docling-vision/users/said/nanoVLM/GraniteDoclingV1-stage1-init/nanoVLM_siglip2-base-patch16-512_2048_mp4_GraniteDoclingV1-stage1-init_16xGPU_full_ds_bs64_287500_lr_vision_0.002-language_0.002-0.0212_0121-140230_lsf394476/converted_untied/step_107000 \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
  --batch-size 64
  
  
  
  For Finetuned Omniparser multi-dataset evaluation (path /proj/docling-vision/users/said/webshot-dataset/runs/detect/omniparser_55c_ft/weights/best.pt):
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
    --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
    --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
    --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
    --yolo-model /proj/docling-vision/users/said/webshot-dataset/runs/detect/omniparser_55c_ft/weights/best.pt \
    --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
    --batch-size 64
    
    
    
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=web,class_schema=screenspot \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=screenspot \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=screenspot \
    --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
    --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=100,class_schema=custom55,tag=yolo100 \
    --dataset format=yolo,image_dir=/proj/docling-vision/users/said/data/yolo_filtered/images/test,labels_dir=/proj/docling-vision/users/said/data/yolo_filtered/labels/test,classes=/proj/docling-vision/users/said/data/yolo_filtered/classes.txt,max_samples=1000,class_schema=custom55,tag=yolo1000 \
    --rtdetr /proj/docling-vision/users/said/webshot-dataset/rtdetrv2.torchscript.pt \
    --rtdetr-classes /proj/docling-vision/users/said/webshot-dataset/coco_classes.txt \
    --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
    --batch-size 64
    
    
Evaluation of GroundCUA dataset (for not only 870 samples but whole dataset) for each models:
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=4350,class_schema=groundcua \
  --screenvlm /proj/docling-vision/users/said/nanoVLM/GraniteDoclingV1-stage1-init/nanoVLM_siglip2-base-patch16-512_2048_mp4_GraniteDoclingV1-stage1-init_16xGPU_full_ds_bs64_287500_lr_vision_0.002-language_0.002-0.0212_0109-164905_lsf361208/converted_untied/step_150000 \
  --class-schema groundcua \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall,recall_agnostic,ned \
  --batch-size 64
  

PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=4350,class_schema=groundcua \
  --rtdetr /proj/docling-vision/users/said/RTDETRv2/outputs_rtdetrv2_r50vd_6x_coco/model_gpu.torchscript.pt \
  --rtdetr-classes /proj/docling-vision/users/said/data/yolo_filtered/classes.txt \
  --class-schema groundcua \
  --metrics page_iou,page_iou_recall,label_page_iou,map,recall,recall_agnostic,ned \
  --batch-size 64
  
  

Visualization only (no evaluation) examples:

Recreate GT vs pred side‑by‑side with thicker borders + new colors:
  PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
  --offline-pred mymodel=/proj/docling-vision/users/said/webshot-dataset/evaluation/m1-step-150000_n864_ds-gcua_map-gcua/preds/step_150000 \
  --viz-only \
  --viz-line-width 4 \
  --viz-gt-color "#00a000" \
  --viz-pred-color "255,128,0"


Prediction‑only overlay:
  PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=groundcua,root=/path/to/GroundCUA,class_schema=groundcua \
  --offline-pred mymodel=/path/to/preds/mymodel \
  --viz-only \
  --viz-include pred \
  --viz-layout overlay
  
/proj/docling-vision/users/said/webshot-dataset/evaluation/m1-omniparser-ft-6_n864_ds-gcua_map-gcua/preds/omniparser_ft_6
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=groundcua,root=/proj/docling-vision/users/said/GroundCUA/GroundCUA,max_samples=870,class_schema=groundcua \
  --offline-pred mymodel=/proj/docling-vision/users/said/webshot-dataset/evaluation/m1-omniparser-ft-6_n864_ds-gcua_map-gcua/preds/omniparser_ft_6 \
  --viz-only



/proj/docling-vision/users/said/webshot-dataset/evaluation/m1-omniparser-ft-6_n210_ds-ss-pc_map-c55/preds/omniparser_ft_6
--dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=custom55 \
/proj/docling-vision/users/said/webshot-dataset/evaluation/m1-omniparser-ft-6_n864_ds-gcua_map-gcua/preds/omniparser_ft_6
PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=pc,class_schema=custom55 \
  --offline-pred mymodel=/proj/docling-vision/users/said/webshot-dataset/evaluation/m1-omniparser-ft-6_n210_ds-ss-pc_map-c55/preds/omniparser_ft_6 \
  --viz-only \
  --viz-labels


PYTHONPATH=src .venv/bin/python -m evaluation.cli \
    --dataset format=screenspot,root=/proj/docling-vision/users/said/SeeClick/ScreenSpot,split=mobile,class_schema=custom55 \
    --yolo-model /proj/docling-vision/users/said/webshot-dataset/runs/detect/omniparser_55c_ft/weights/omniparser_ft_6.pt \
    --metrics page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned \
    --batch-size 64 \
    --viz-labels

  """


from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

from .datasets import (
    build_raw_dataset,
    build_yolo_dataset,
    build_groundcua_dataset,
    build_screenspot_dataset,
    load_elements_file,
)
from .label_mapping import LabelMapper
from .metrics.label_page_iou import LabelAwarePageIoU
from .metrics.map import MeanAveragePrecision
from .metrics.recall import Recall
from .metrics.ned import NormalizedEditDistance
from .metrics.page_iou import PageIoU, PageIoURecall
from .models.base import OfflinePredictionRunner
from .models.gemini import GeminiRunner
from .models.paddleocrvl import PaddleOCRVLRunner
from .models.qwen3_vl import Qwen3VLRunner
from .models.internvl3 import InternVL3Runner
from .models.rtdetr import RTDETRModelRunner
from .models.screenvlm import ScreenVLMRunner
from .models.yolo import YoloModelRunner
from .runner import Evaluator
from .viz import VizConfig, legacy_defaults, save_viz


def _metric_specs(metric_names: List[str], args) -> List[Tuple[str, Dict]]:
    specs = []
    for name in metric_names:
        if name == "page_iou":
            specs.append(("page_iou", {"max_resolution": args.pageiou_resolution}))
        elif name in ("page_iou_recall", "pageiou_recall", "page_iou_rec"):
            specs.append(("page_iou_recall", {"max_resolution": args.pageiou_resolution}))
        elif name in ("label_page_iou", "labelawarepageiou"):
            specs.append(("label_page_iou", {"max_resolution": args.pageiou_resolution}))
        elif name in ("map", "map_50", "map50"):
            specs.append(("map", {"iou_threshold": args.map_iou_thr}))
        elif name in ("recall", "recall_label", "recall_labelaware"):
            specs.append(("recall", {"iou_threshold": args.recall_iou_thr, "label_aware": True}))
        elif name in ("recall_agnostic", "recall_nolabel", "recall_any"):
            specs.append(("recall", {"iou_threshold": args.recall_iou_thr, "label_aware": False}))
        elif name in ("ned", "normalized_edit_distance"):
            specs.append(("ned", {"iou_threshold": args.ned_iou_thr}))
        else:
            print(f"Warning: unknown metric '{name}' - skipping.")
    return specs


def _build_metrics(specs: List[Tuple[str, Dict]]):
    metrics = []
    for name, kwargs in specs:
        if name == "page_iou":
            metrics.append(PageIoU(**kwargs))
        elif name == "page_iou_recall":
            metrics.append(PageIoURecall(**kwargs))
        elif name == "label_page_iou":
            metrics.append(LabelAwarePageIoU(**kwargs))
        elif name == "map":
            metrics.append(MeanAveragePrecision(**kwargs))
        elif name == "recall":
            metrics.append(Recall(**kwargs))
        elif name == "ned":
            metrics.append(NormalizedEditDistance(**kwargs))
    return metrics


def _slug(text: str, max_len: int = 24) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        return "unknown"
    return cleaned[:max_len]


def _models_token(model_names: List[str]) -> str:
    names = [_slug(n, 20) for n in model_names if n]
    if not names:
        return "model"
    if len(names) <= 5:
        return "+".join(names)
    return "+".join(names[:5]) + f"+{len(names)-5}"


def _dataset_token(args) -> str:
    if args.groundcua_root:
        base = _slug(Path(args.groundcua_root).name, 16)
        if "groundcua" in base:
            base = "gcua"
        else:
            base = f"gcua-{base}"
    elif args.screenspot_root:
        base = _slug(Path(args.screenspot_root).name, 16)
        split = args.screenspot_split.lower()
        if split != "all":
            base = f"ss-{split}"
        elif "screenspot" in base:
            base = "ss"
        else:
            base = f"ss-{base}"
    elif args.format == "yolo" or args.labels_dir:
        base = f"yolo-{_slug(Path(args.image_dir).name, 16)}"
    else:
        base = f"raw-{_slug(Path(args.image_dir).name, 16)}"
    tag = getattr(args, "dataset_tag", None)
    if tag:
        base = f"{base}-{_slug(tag, 8)}"
    return base


def _schema_token(schema: str) -> str:
    if schema == "groundcua":
        return "gcua"
    if schema == "screenspot":
        return "ss"
    return "c55"


def _resolve_output_paths(args, dataset_len: int, model_names: List[str]) -> Tuple[Path, Path]:
    models_tok = _models_token(model_names)
    dataset_tok = _dataset_token(args)
    schema_tok = _schema_token(args.class_schema)
    run_id = f"m{len(model_names)}-{models_tok}_n{dataset_len}_ds-{dataset_tok}_map-{schema_tok}"
    base_dir = Path("evaluation") / run_id
    out_path = Path(args.output) if args.output else base_dir / "report.json"
    preds_dir = Path(args.save_preds) if args.save_preds else base_dir / "preds"
    return out_path, preds_dir


def _parse_color(raw: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if not raw:
        return fallback
    text = raw.strip().lower()
    if text.startswith("#"):
        text = text[1:]
    if text.startswith("0x"):
        text = text[2:]
    if len(text) == 6 and all(c in "0123456789abcdef" for c in text):
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
        return (r, g, b)
    parts = [p for p in re.split(r"[ ,]+", text) if p]
    if len(parts) != 3:
        return fallback
    try:
        vals = [int(float(p)) for p in parts]
    except ValueError:
        return fallback
    vals = [max(0, min(255, v)) for v in vals]
    return (vals[0], vals[1], vals[2])


def _build_viz_config(args) -> VizConfig:
    base = legacy_defaults() if args.viz_style == "legacy" else VizConfig()
    include_gt = args.viz_include in ("both", "gt")
    include_pred = args.viz_include in ("both", "pred")
    line_width = args.viz_line_width if args.viz_line_width is not None else base.line_width
    label_max = args.viz_label_max if args.viz_label_max is not None else base.label_max_len
    return VizConfig(
        include_gt=include_gt,
        include_pred=include_pred,
        layout=args.viz_layout,
        style=args.viz_style,
        gt_color=_parse_color(args.viz_gt_color, base.gt_color),
        pred_color=_parse_color(args.viz_pred_color, base.pred_color),
        line_width=line_width,
        label_max_len=label_max,
        fill_alpha=base.fill_alpha,
        halo_color=base.halo_color,
        halo_width=base.halo_width,
        label_font_size=base.label_font_size,
        label_pad=base.label_pad,
        label_bg_alpha=base.label_bg_alpha,
        label_text_color=base.label_text_color,
        title_font_size=base.title_font_size,
        title_pad=base.title_pad,
        title_bg_alpha=base.title_bg_alpha,
        title_text_color=base.title_text_color,
        show_panel_titles=base.show_panel_titles,
        divider_color=base.divider_color,
        divider_width=base.divider_width,
        show_labels=args.viz_labels,
        jpeg_quality=base.jpeg_quality,
    )


def _prediction_path(sample: "EvaluationSample", pred_dir: Path, suffix: str) -> Path:
    stem = sample.sample_id or Path(sample.image_path).stem
    return pred_dir / f"{stem}{suffix}"


def _run_viz_only(dataset_runs, args):
    config = _build_viz_config(args)
    if not config.include_gt and not config.include_pred:
        raise SystemExit("--viz-include must include at least one of gt or pred.")

    pred_specs: List[Tuple[str, str]] = []
    for entry in args.offline_pred:
        name, path = _parse_name_path(entry)
        pred_specs.append((name, path))

    if config.include_pred and not pred_specs:
        raise SystemExit("--viz-only with pred visualizations requires --offline-pred.")
    if not pred_specs and not args.viz_out:
        raise SystemExit("--viz-only without --offline-pred needs --viz-out to write images.")

    pred_suffix = args.pred_suffix
    for ds_args, dataset, schema in dataset_runs:
        dataset_tok = _dataset_token(ds_args)
        print(f"\n[Viz] {dataset_tok}: {len(dataset)} samples, schema={schema}")
        if pred_specs:
            for name, path in pred_specs:
                pred_dir = Path(path)
                out_root = Path(args.viz_out) if args.viz_out else pred_dir
                if args.viz_out and len(pred_specs) > 1:
                    out_root = out_root / name
                written = 0
                missing = 0
                for sample in tqdm(dataset, desc=f"  {name}"):
                    preds = []
                    if config.include_pred:
                        pred_path = _prediction_path(sample, pred_dir, pred_suffix)
                        if pred_path.exists():
                            preds = load_elements_file(pred_path, include_raw=False)
                        else:
                            missing += 1
                            if not config.include_gt:
                                continue
                    out_path = save_viz(out_root, sample, preds, config, suffix=args.viz_suffix)
                    if out_path:
                        written += 1
                print(f"  {name}: wrote {written} viz files ({missing} missing preds) to {out_root.resolve()}.")
        else:
            out_root = Path(args.viz_out)
            written = 0
            for sample in dataset:
                out_path = save_viz(out_root, sample, [], config, suffix=args.viz_suffix)
                if out_path:
                    written += 1
            print(f"  wrote {written} viz files to {out_root.resolve()}.")


def _build_runner_from_spec(spec: Tuple[str, Dict]):
    kind, kwargs = spec
    if kind == "yolo":
        return YoloModelRunner(**kwargs)
    if kind == "offline":
        return OfflinePredictionRunner(**kwargs)
    if kind == "qwen3_vl":
        return Qwen3VLRunner(**kwargs)
    if kind == "internvl3":
        return InternVL3Runner(**kwargs)
    if kind == "gemini":
        return GeminiRunner(**kwargs)
    if kind == "paddleocrvl":
        return PaddleOCRVLRunner(**kwargs)
    if kind == "screenvlm":
        return ScreenVLMRunner(**kwargs)
    if kind == "rtdetr":
        return RTDETRModelRunner(**kwargs)
    raise ValueError(f"Unknown model kind {kind}")


def _eval_worker(dataset, metric_specs, model_spec, pred_dir, batch_size, queue, class_schema, save_viz):
    try:
        metrics = _build_metrics(metric_specs)
        mapper = LabelMapper(class_schema)
        evaluator = Evaluator(metrics, label_mapper=mapper)
        model = _build_runner_from_spec(model_spec)
        report = evaluator.evaluate_model(
            dataset,
            model,
            save_predictions_dir=pred_dir,
            batch_size=batch_size,
            save_viz=save_viz,
        )
        queue.put({"ok": True, "report": report})
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc)})


def _parse_name_path(val: str) -> Tuple[str, str]:
    if "=" in val:
        name, path = val.split("=", 1)
        return name.strip(), path.strip()
    path = val.strip()
    return Path(path).stem, path


def _parse_dataset_spec(spec: str) -> Dict[str, str]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("Dataset spec is empty.")
    out: Dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid dataset spec segment: {part!r}. Use key=value pairs.")
        key, val = part.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def _apply_dataset_spec(base_args, spec: Dict[str, str]):
    data = vars(base_args).copy()
    # Reset dataset-specific fields to avoid leaking base dataset values.
    data.update(
        {
            "image_dir": None,
            "labels_dir": None,
            "classes": None,
            "format": "auto",
            "groundcua_root": None,
            "screenspot_root": None,
            "screenspot_split": "all",
            "max_samples": None,
            "dataset_tag": None,
        }
    )

    alias_map = {
        "schema": "class_schema",
        "class": "class_schema",
        "max": "max_samples",
        "samples": "max_samples",
        "split": "screenspot_split",
        "name": "dataset_tag",
        "tag": "dataset_tag",
        "type": "format",
        "dataset": "format",
        "root": "root",
        "path": "root",
    }

    normalized: Dict[str, str] = {}
    for key, val in spec.items():
        key_lower = key.strip().lower()
        key_lower = alias_map.get(key_lower, key_lower)
        normalized[key_lower] = val

    fmt = normalized.get("format")
    if fmt:
        fmt_lower = fmt.strip().lower()
        if fmt_lower in ("groundcua", "gcua"):
            data["format"] = "auto"
            root = normalized.get("groundcua_root") or normalized.get("root")
            if not root:
                raise ValueError("GroundCUA dataset requires groundcua_root or root.")
            data["groundcua_root"] = root
        elif fmt_lower in ("screenspot", "ss"):
            data["format"] = "auto"
            root = normalized.get("screenspot_root") or normalized.get("root")
            if not root:
                raise ValueError("ScreenSpot dataset requires screenspot_root or root.")
            data["screenspot_root"] = root
        else:
            data["format"] = fmt_lower

    if "image_dir" in normalized:
        data["image_dir"] = normalized["image_dir"]
    if "labels_dir" in normalized:
        data["labels_dir"] = normalized["labels_dir"]
    if "classes" in normalized:
        data["classes"] = normalized["classes"]
    if "groundcua_root" in normalized:
        data["groundcua_root"] = normalized["groundcua_root"]
    if "screenspot_root" in normalized:
        data["screenspot_root"] = normalized["screenspot_root"]
    if "screenspot_split" in normalized:
        data["screenspot_split"] = normalized["screenspot_split"].lower()
    if "class_schema" in normalized:
        data["class_schema"] = normalized["class_schema"]
    if "dataset_tag" in normalized:
        data["dataset_tag"] = normalized["dataset_tag"]
    if "max_samples" in normalized:
        raw = normalized["max_samples"].strip()
        if not raw or raw.lower() in ("none", "null"):
            data["max_samples"] = None
        else:
            data["max_samples"] = int(raw)

    # Allow root alias without explicit format.
    if data["groundcua_root"]:
        data["screenspot_root"] = None
    if data["screenspot_root"]:
        data["groundcua_root"] = None

    return SimpleNamespace(**data)


def _validate_dataset_args(args):
    if args.screenspot_root or args.groundcua_root:
        return
    if not args.image_dir:
        raise ValueError("Dataset spec must include image_dir or a dataset root.")


def _build_model_specs(args, class_schema: str) -> List[Tuple[str, Dict]]:
    model_specs: List[Tuple[str, Dict]] = []

    for entry in args.offline_pred:
        name, path = _parse_name_path(entry)
        model_specs.append(
            ("offline", dict(name=name, predictions_dir=path, suffix=args.pred_suffix))
        )

    for entry in args.yolo_model:
        name, path = _parse_name_path(entry)
        model_specs.append(
            (
                "yolo",
                dict(
                    weights_path=path,
                    name=name,
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                    imgsz=args.yolo_imgsz,
                    device=args.device,
                ),
            )
        )

    if args.omniparser_weights:
        model_specs.append(
            (
                "yolo",
                dict(
                    weights_path=args.omniparser_weights,
                    name="omniparser",
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                    imgsz=args.yolo_imgsz,
                    device=args.device,
                ),
            )
        )

    for entry in args.screenvlm:
        name, checkpoint = _parse_name_path(entry)
        model_specs.append(
            (
                "screenvlm",
                dict(
                    checkpoint=checkpoint,
                    name=name,
                    max_new_tokens=args.screenvlm_max_new_tokens,
                    temperature=args.screenvlm_temperature,
                    top_p=args.screenvlm_top_p,
                    top_k=args.screenvlm_top_k,
                ),
            )
        )

    for entry in args.qwen3_vl:
        name, model_id = _parse_name_path(entry)
        model_specs.append(
            (
                "qwen3_vl",
                dict(
                    model_id=model_id,
                    name=name,
                    prompt=args.qwen_prompt,
                    max_new_tokens=args.qwen_max_new_tokens,
                    temperature=args.qwen_temperature,
                    top_p=args.qwen_top_p,
                    class_schema=class_schema,
                ),
            )
        )

    for entry in args.internvl3:
        name, model_id = _parse_name_path(entry)
        model_specs.append(
            (
                "internvl3",
                dict(
                    model_id=model_id,
                    name=name,
                    prompt=args.internvl3_prompt,
                    max_new_tokens=args.internvl3_max_new_tokens,
                    temperature=args.internvl3_temperature,
                    top_p=args.internvl3_top_p,
                    max_model_len=args.internvl3_max_model_len,
                    class_schema=class_schema,
                ),
            )
        )

    for entry in args.gemini:
        name, model_id = _parse_name_path(entry)
        model_specs.append(
            (
                "gemini",
                dict(
                    model_id=model_id,
                    name=name,
                    api_key=args.gemini_api_key,
                    prompt=args.qwen_prompt,
                    class_schema=class_schema,
                ),
            )
        )

    if args.paddle_ocrvl:
        model_specs.append(
            (
                "paddleocrvl",
                dict(
                    python_path=args.paddle_python,
                    name="paddleocrvl",
                ),
            )
        )

    for entry in args.rtdetr:
        name, path = _parse_name_path(entry)
        model_specs.append(
            (
                "rtdetr",
                dict(
                    model_path=path,
                    name=name,
                    conf=args.rtdetr_conf,
                    device=args.device,
                    classes_file=args.rtdetr_classes,
                    input_size=tuple(args.rtdetr_input_size),
                ),
            )
        )

    if not model_specs:
        raise SystemExit("No models specified. Use --yolo-model or --offline-pred.")

    return model_specs


def _build_dataset(args, label_mapper):
    fmt = args.format
    if args.screenspot_root:
        return build_screenspot_dataset(
            root_dir=args.screenspot_root,
            split=args.screenspot_split,
            max_samples=args.max_samples,
            label_mapper=label_mapper,
        )
    if args.groundcua_root:
        return build_groundcua_dataset(
            root_dir=args.groundcua_root,
            max_samples=args.max_samples,
            label_mapper=label_mapper,
        )

    if fmt == "auto":
        fmt = "yolo" if args.labels_dir else "raw"

    if fmt == "yolo":
        return build_yolo_dataset(
            image_dir=args.image_dir,
            labels_dir=args.labels_dir,
            classes_path=args.classes,
            max_samples=args.max_samples,
            label_mapper=label_mapper,
        )
    return build_raw_dataset(image_dir=args.image_dir, max_samples=args.max_samples, label_mapper=label_mapper)


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate screen parsing models.")
    parser.add_argument("--image-dir", help="Directory with evaluation images.")
    parser.add_argument("--labels-dir", help="YOLO labels directory (only for --format yolo/auto).")
    parser.add_argument("--classes", help="Optional classes.txt for YOLO labels.")
    parser.add_argument("--format", choices=["auto", "raw", "yolo"], default="auto", help="Dataset format.")
    parser.add_argument("--groundcua-root", help="GroundCUA root directory (contains data/ and images/).")
    parser.add_argument("--screenspot-root", help="ScreenSpot root directory (contains data/ and images/).")
    parser.add_argument(
        "--screenspot-split",
        choices=["all", "web", "pc", "mobile"],
        default="all",
        help="ScreenSpot split to evaluate.",
    )
    parser.add_argument("--max-samples", type=int, help="Cap number of samples for quick runs.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Repeatable dataset spec (key=value, comma-separated). Example: "
            "format=yolo,image_dir=/path,labels_dir=/path,max_samples=100,tag=yolo100 "
            "or format=groundcua,root=/path,max_samples=870,class_schema=groundcua "
            "or format=screenspot,root=/path,split=web,max_samples=200,class_schema=screenspot."
        ),
    )

    parser.add_argument("--yolo-model", action="append", default=[], help="YOLO weights (name=path or just path).")
    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--yolo-iou", type=float, default=0.10)
    parser.add_argument("--yolo-imgsz", type=int, default=1280)
    parser.add_argument("--device", help="Optional inference device hint.")
    parser.add_argument(
        "--omniparser-weights",
        help="Path to OmniParser YOLOv8 weights (treated as a YOLO model).",
    )

    parser.add_argument(
        "--qwen3-vl",
        action="append",
        default=[],
        help="Use Qwen/Qwen3-VL-8B-Instruct via vLLM (name=model_id or just model_id).",
    )
    parser.add_argument("--qwen-prompt", help="Custom prompt for Qwen3-VL.")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=8192)
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-top-p", type=float, default=0.9)
    parser.add_argument(
        "--internvl3",
        action="append",
        default=[],
        help="Use OpenGVLab/InternVL3-2B via vLLM (name=path_or_id or just path_or_id).",
    )
    parser.add_argument("--internvl3-prompt", help="Custom prompt for InternVL3.")
    parser.add_argument("--internvl3-max-new-tokens", type=int, default=8192)
    parser.add_argument("--internvl3-temperature", type=float, default=0.0)
    parser.add_argument("--internvl3-top-p", type=float, default=0.9)
    parser.add_argument("--internvl3-max-model-len", type=int, default=8192)
    parser.add_argument(
        "--screenvlm",
        action="append",
        default=[],
        help="ScreenVLM checkpoint (name=path or just path).",
    )
    parser.add_argument("--screenvlm-max-new-tokens", type=int, default=6192)
    parser.add_argument("--screenvlm-temperature", type=float, default=0.0)
    parser.add_argument("--screenvlm-top-p", type=float, default=0.9)
    parser.add_argument("--screenvlm-top-k", type=int, default=50)
    parser.add_argument(
        "--gemini",
        action="append",
        default=[],
        help="Use Gemini via google-genai (name=model_id or just model_id).",
    )
    parser.add_argument("--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY/GOOGLE_API_KEY).")
    parser.add_argument(
        "--paddle-ocrvl",
        action="store_true",
        help="Use PaddleOCRVL via separate Python env (.venv-paddle/bin/python).",
    )
    parser.add_argument(
        "--paddle-python",
        default=".venv-paddle/bin/python",
        help="Python executable for PaddleOCRVL.",
    )
    parser.add_argument(
        "--paddle-bridge",
        default=str((Path(__file__).resolve().parent / "models" / "paddle_ocrvl_bridge.py")),
        help="Bridge script path for PaddleOCRVL.",
    )

    # RT-DETR model arguments (uses TorchScript exported model)
    parser.add_argument(
        "--rtdetr",
        action="append",
        default=[],
        help="RT-DETR TorchScript model path (name=path or just path to .pt file).",
    )
    parser.add_argument("--rtdetr-conf", type=float, default=0.40, help="RT-DETR confidence threshold.")
    parser.add_argument(
        "--rtdetr-classes",
        help="Path to classes file (.txt/.json/.yaml) for RT-DETR label mapping.",
    )
    parser.add_argument(
        "--rtdetr-input-size",
        type=int,
        nargs=2,
        default=[736, 1280],
        metavar=("H", "W"),
        help="RT-DETR input size (height width).",
    )

    parser.add_argument(
        "--offline-pred",
        action="append",
        default=[],
        help="Directory with precomputed predictions (name=dir or just dir).",
    )
    parser.add_argument(
        "--pred-suffix",
        default=".pred.json",
        help="Prediction file suffix for offline predictions (default: .pred.json).",
    )

    parser.add_argument("--pageiou-resolution", type=int, default=0, help="Downscale long side before PageIoU.")
    parser.add_argument("--map-iou-thr", type=float, default=0.5, help="IoU threshold for mAP.")
    parser.add_argument("--recall-iou-thr", type=float, default=0.5, help="IoU threshold for recall.")
    parser.add_argument("--ned-iou-thr", type=float, default=0.0, help="IoU threshold for NED matching.")
    parser.add_argument(
        "--metrics",
        default="page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned",
        help="Comma-separated metrics: page_iou,page_iou_recall,label_page_iou,map,recall_label,recall_agnostic,ned",
    )
    parser.add_argument(
        "--class-schema",
        choices=["custom55", "groundcua", "screenspot"],
        default="custom55",
        help="Target class schema for labels.",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size for predict_batch.")
    parser.add_argument("--save-preds", help="Base directory to dump model predictions.")
    parser.add_argument("--output", help="Path to write JSON report.")
    parser.add_argument("--no-viz", action="store_true", help="Disable GT vs prediction visualization output.")
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Render visualizations from existing predictions without running evaluation.",
    )
    parser.add_argument(
        "--viz-include",
        choices=["both", "gt", "pred"],
        default="both",
        help="Which annotations to render in viz-only mode.",
    )
    parser.add_argument(
        "--viz-style",
        choices=["modern", "legacy"],
        default="legacy",
        help="Visualization style preset (legacy matches the original simple boxes).",
    )
    parser.add_argument(
        "--viz-layout",
        choices=["side_by_side", "overlay"],
        default="side_by_side",
        help="Viz layout for gt+pred rendering.",
    )
    parser.add_argument(
        "--viz-gt-color",
        default=None,
        help="Ground-truth box color as R,G,B or hex (e.g. 0,160,0 or #00a000).",
    )
    parser.add_argument(
        "--viz-pred-color",
        default=None,
        help="Prediction box color as R,G,B or hex (e.g. 200,30,30 or #c81e1e).",
    )
    parser.add_argument("--viz-line-width", type=int, default=None, help="Bounding box line width.")
    parser.add_argument("--viz-label-max", type=int, default=None, help="Max label length to render.")
    parser.add_argument(
        "--viz-labels",
        action="store_true",
        help="Render label text on boxes (off by default to reduce clutter).",
    )
    parser.add_argument(
        "--viz-suffix",
        default=".viz.jpg",
        help="Filename suffix for viz outputs.",
    )
    parser.add_argument(
        "--viz-out",
        help="Optional output directory for viz-only (defaults to predictions dir).",
    )

    args = parser.parse_args(argv)

    dataset_runs: List[Tuple[SimpleNamespace, List, str]] = []
    if args.dataset:
        for raw_spec in args.dataset:
            try:
                spec = _parse_dataset_spec(raw_spec)
                ds_args = _apply_dataset_spec(args, spec)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            ds_args.output = None
            ds_args.save_preds = None
            _validate_dataset_args(ds_args)
            schema = ds_args.class_schema
            mapper = LabelMapper(schema)
            dataset = _build_dataset(ds_args, label_mapper=mapper)
            if not dataset:
                raise SystemExit(f"No evaluation samples found for dataset spec: {raw_spec}")
            dataset_runs.append((ds_args, dataset, schema))
    else:
        _validate_dataset_args(args)
        mapper = LabelMapper(args.class_schema)
        dataset = _build_dataset(args, label_mapper=mapper)
        if not dataset:
            raise SystemExit("No evaluation samples found for given paths.")
        dataset_runs.append((SimpleNamespace(**vars(args)), dataset, args.class_schema))

    if args.viz_only:
        _run_viz_only(dataset_runs, args)
        return

    metric_names = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    metric_specs = _metric_specs(metric_names, args)
    if not metric_specs:
        raise SystemExit("No valid metrics selected.")

    ctx = mp.get_context("spawn")
    for ds_args, dataset, schema in dataset_runs:
        model_specs = _build_model_specs(args, schema)
        model_names = [cfg.get("name") or cfg.get("model_id") or kind for kind, cfg in model_specs]
        dataset_tok = _dataset_token(ds_args)
        print(f"\n[Dataset] {dataset_tok}: {len(dataset)} samples, schema={schema}")
        print(f"Running {len(model_specs)} model(s): {model_names}")

        summary_rows = []
        out_path, base_pred_dir = _resolve_output_paths(ds_args, len(dataset), model_names)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for kind, cfg in model_specs:
            model_name = cfg.get("name") or cfg.get("model_id") or kind
            pred_dir = base_pred_dir / model_name
            print(f"\n[Model] Starting isolated process for {model_name}")
            queue = ctx.Queue()
            proc = ctx.Process(
                target=_eval_worker,
                args=(
                    dataset,
                    metric_specs,
                    (kind, cfg),
                    str(pred_dir) if pred_dir else None,
                    args.batch_size,
                    queue,
                    schema,
                    not args.no_viz,
                ),
            )
            proc.start()
            result = queue.get()
            proc.join()

            if not result.get("ok"):
                err = result.get("error", "Unknown error")
                raise SystemExit(f"Model {model_name} failed: {err}")

            report = result["report"]
            print(f"Model {model_name}:")
            for metric_name, data in report["dataset_metrics"].items():
                print(f"  {metric_name}: {data.get('value')}")
            summary_rows.append((model_name, report["dataset_metrics"]))

            existing = []
            if out_path.exists():
                with out_path.open("r", encoding="utf-8") as f:
                    try:
                        loaded = json.load(f)
                        existing = loaded if isinstance(loaded, list) else [loaded]
                    except json.JSONDecodeError:
                        existing = []
            merged = existing + [report]
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            print(f"  Appended report to {out_path}")

        if summary_rows:
            import csv
            csv_path = out_path.parent / "summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            metric_names = sorted({m for _, metrics in summary_rows for m in metrics.keys()})
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["model"] + metric_names)
                for model_name, metrics in summary_rows:
                    row = [model_name]
                    for m in metric_names:
                        val = metrics.get(m, {}).get("value")
                        if isinstance(val, float):
                            row.append(f"{val:.3f}")
                        elif val is None:
                            row.append("")
                        else:
                            row.append(val)
                    writer.writerow(row)
            print(f"\nWrote summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
