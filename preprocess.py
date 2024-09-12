# %%
import os # para conexão com o windows e criacao de pastas
import shutil # para alterações em pastas
from glob import Path # transformar em lista
from pathlib import Path
from collections import Counter 
from sklearn.model_selection import train_test_split
import cv2
import yaml

def preprocess_dataset(imgs_path=None, txts_path=None,videos_path=None, classes_txt=None, yolo_ds_path=None,val_size=0.2, img_filetypes=("jpeg", "jpg", "png", "tif", "bmp"), frame_rate=1, 
                       random_state=42):
    """
    Pré Processamento do Dataset para Yolov9m, esssa função vai:
    1. Carregar arquivos de images e/ou videos e extrair os frames de cada video
    2. Garante que cada imagem ou cada frame extraido tenha seu respectivo txt
    3. Separa o database entre treino e validação
    4. Prepara a estrutura de dataset para o Yolov9m
    5. Cria um dataset yaml file que precisará para o treino do Yolov9
    
    Parametros (Foi necessário colocar none em todos como padrão porque as operações em que usaremos essa função podem ser só imagens ou só videos):
    - IMGS_PATH (Path ou String): path para o diretório que contém as imagens.
    - TXTS_PATH (Path ou String): path para o diretório contendo txt
    - VIDEOS_PATH (Path ou String): path para o diretório contendo os videos
    - CLASSES_TXT (Path ou String): path para o arquivo classes_txt
    - YOLO_DS_PATH (path ou string): path onde o dataset pronto do yolo será salvo
    - val_size (Float): Porcentagem de imagens que serão usadas na validação. Default como 0.2 (20%)
    - img_filetypes (tuple): Tupla de imagens contendo todas as extensões que serão buscadas
    - frame_rate (int): Frame rate para extrair os frames dos videos (em frames por segundo)
    - random_state (int): Semente para reprodução
    
    Retorna:
    - Nada
    """
    
    if not imgs_path and not videos_path:
        raise ValueError("Dê pelo menos uma pasta de video ou pelo menos uma pasta de imagens")
    
    # Processamento inicial
    
    imgs = []

    # 1. Carrega Imagens
    if imgs_path:
        imgs_path = Path(imgs_path) # nao se esqueça de colocar o r
        for filetype in img_filetypes:
            imgs.extend(imgs_path.glob(f"*.{filetype}"))
    
    # 2. Extrai frames dos videos
    if videos_path:
        videos_path = Path(videos_path)
        videos_files = list(videos_path.glob("*.mp4"))
        
        def extract_frames(video_file, output_dir, frame_rate=1):
            """Extrai os frames de cada video baseado no frame_rate especificado"""
            video_name = video_file.stem
            cap = cv2.VideoCapture(str(video_file))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) # fps do video
            count = 0
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if count % (fps // frame_rate) == 0: # Captura um frame de um especifico frame rate
                    frame_name = f"{video_name}_frame_{frame_count}.jpg" # Nome do frame retirado
                    frame_path = output_dir / frame_name
                    cv2.imwrite(str(frame_path) , frame)
                    imgs.append(frame_path)
                    frame_count += 1
                count += 1
            cap.release()
            
        # Define um diretorio temporario para guardar os frames extraidos
        
        extracted_frames_dir = Path("extracted_frames")
        extracted_frames_dir.mkdir(exist_ok=True)
        
        for video in videos_files:
            extract_frames(video,extracted_frames_dir,frame_rate)
    
    # 3. Carrega txts
    if txts_path:
        txts_path = Path(txts_path)
        txts = list(txts_path.glob("*.txts"))
    else:
        txts = []
        
    # Garantir que haja consistência entre imagens e txts
    
    img_stems = (img.stem for img in imgs)
    txt_stems = (txt.stem for txt in txts)
    
    # Mantem somente os pares onde imagem e txt existem
    
    imgs = [img for img in imgs if img.stem in txt_stems]
    txts = [txt for txt in txts if txt.stem in img_stems] 
    
    if len(imgs) == 0 or len(txts) == 0:
        raise ValueError("Nenhum par válido foi encontrado. Cheque o caminho e os formatos")
        
    imgs.sort(key=lambda x: x.stem)
    txts.sort(key=lambda x: x.stem)
    
    # 4. Carrega o nome das clases
    with open(classes_txt, "r") as f:
        classes = f.read().splitlines()
    
    # 5. Preparar as Labels para cada imagem (Usado no Stratificado)
    labels = []
    for txt in txts:
        with open(txt,"r") as f:
            label_lines = f.readlines()
            labels_in_file = [line.split()[0] for line in label_lines]
            labels.append(labels_in_file)
    
    # Comprimir / Achatar as labels em uma listas para contagem
    flat_labels = [labels for sublist in labels for label in sublist]
    
    # 6. Separacao entre Treino e Validação  
    imgs_train , imgs_valid , txts_train, txts_valid = train_test_split(
        imgs, txts, test_size=val_size , random_state=random_state, stratify=flat_labels
        )           
    
    # 7. Criar a Pasta do Yolo
    def create_yolo_dirs(base_path):
        labels_dir = base_path/"labels"
        images_dir = base_path/"images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        return labels_dir, images_dir
    
    # Diretorios para Treino e Validação
    
    train_labels_dir, train_images_dir = create_yolo_dirs(yolo_ds_path / "train")
    valid_labels_dir, valid_images_dir = create_yolo_dirs(yolo_ds_path / "valid") 
    
    # 8. Copiando as imagens e os labels de treino e testes para os diretorios criados
    
    def copy_files(imgs,txts,imgs_dst_dir, txt_dst_dir):
        for img, txt in zip(imgs,txts):
            shutil.copy(img,imgs_dst_dir / img.name)
            shutil.copy(txt,txt_dst_dir / txt.name)
    
    copy_files(imgs_train, txts_train , train_images_dir, train_labels_dir)
    copy_files(imgs_valid , txts_valid , valid_images_dir, valid_labels_dir)
    
    # 9. Criação do YAML para o YOLO

    yaml_data = {
        "names": classes,
        "nc": len(classes),
        "train": str(yolo_ds_path / "train"),
        "val": str(yolo_ds_path / "valid"),
    }
    
    yaml_path = yolo_ds_path / "custom_dataset.yaml"
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)
        
    print(f"Preprocessing complete. Dataset ready at: {yolo_ds_path}")
    print(f"YAML file created at: {yaml_path}")

    # Remove o diretorio temporário dos frames
    shutil.rmtree(extracted_frames_dir)        
        
        
        
    

# %%
# Exemplos de Utilização

# Preprocessamento de imagens E videos

# preprocess_dataset(
#     imgs_path=r"C:\Users\isaac\OneDrive\Documentos\Pitayas\ApenasImagens-20240905T202742Z-001\ApenasImagens",
#     txts_path=r"C:\Users\isaac\OneDrive\Documentos\Pitayas\TXTPitaya-20240905T210838Z-001\TXTPitaya",
#     videos_path="path/to/videos",
#     classes_txt=r"C:\Users\isaac\OneDrive\Documentos\Pitayas\classes.txt",
#     yolo_ds_path=r"C:\Users\isaac\OneDrive\Documentos\Pitayas\content\dataset_yolo",
#     val_size=0.2,  # 20% of data for validation
#     frame_rate=1   # Extract 1 frame per second from videos
# )

# Preprocessamento somente de videos

# preprocess_dataset(
#     videos_path="path/to/videos",
#     txts_path="path/to/annotations",
#     classes_txt="path/to/classes.txt",
#     yolo_ds_path="path/to/yolo_dataset",
#     val_size=0.2,  # 20% of data for validation
#     frame_rate=2   # Extract 2 frames per second from videos
# )

# Preprocessamento somente de imagens

# preprocess_dataset(
#     imgs_path="path/to/images",
#     txts_path="path/to/annotations",
#     classes_txt="path/to/classes.txt",
#     yolo_ds_path="path/to/yolo_dataset",
#     val_size=0.2  # 20% of data for validation
# )





