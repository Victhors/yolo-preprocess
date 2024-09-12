### Código de Pré-processamento

Esse repositório introduz uma função de pré-processamento abrangente para lidar com **imagens** e **vídeos** em tarefas de detecção de objetos com o YOLO. A função simplifica o processo de preparação de um dataset ao consolidar o pareamento de anotações de imagens, extração de quadros de vídeos, divisão entre treino e validação, e estruturação do dataset em um único fluxo de trabalho. Abaixo está uma explicação detalhada das principais funcionalidades e do fluxo de trabalho da função de pré-processamento.

---

### Principais Funcionalidades:

1. **Suporte Dual para Imagens e Vídeos**:

   - A função agora lida com datasets que contêm tanto **imagens estáticas** quanto **vídeos**.
   - Se vídeos forem fornecidos, a função extrai quadros de cada vídeo em uma taxa de quadros especificada (por exemplo, 1 quadro por segundo) e trata cada quadro extraído como uma imagem separada para treino e validação.
   - Essa flexibilidade permite que a função trabalhe perfeitamente com datasets contendo apenas imagens, apenas vídeos ou uma mistura de ambos.

2. **Extração de Quadros de Vídeos**:

   - Para o pré-processamento de vídeos, os quadros são extraídos em uma taxa de quadros configurável usando OpenCV. Cada quadro é salvo com um nome único baseado no nome do vídeo e no número do quadro, garantindo que não haja conflitos no dataset de imagens.
   - A função limpa a pasta temporária usada para armazenar os quadros extraídos após o processamento.

3. **Consistência entre Imagens e Anotações**:

   - A função verifica a consistência entre os **arquivos de imagem** (ou quadros de vídeo extraídos) e seus respectivos **arquivos de anotações em texto**. Ela garante que apenas as imagens com anotações correspondentes sejam incluídas no dataset.
   - Quaisquer imagens ou arquivos de anotações não correspondentes são excluídos do dataset final, reduzindo o risco de erros durante o treino.

4. **Divisão entre Treino e Validação**:

   - A função realiza uma **divisão estratificada entre treino e validação** para garantir que a distribuição das classes permaneça equilibrada nos conjuntos de treino e validação. A divisão é personalizável, com uma divisão padrão de 80%-20%.
   - Esta etapa é crucial para evitar o sobreajuste e garantir que o modelo se generalize bem para dados não vistos.

5. **Estrutura do Dataset para YOLO**:

   - A função cria automaticamente a estrutura de pastas necessária para o treino com o YOLOv8, incluindo os subdiretórios `train` e `valid` para **imagens** e **rótulos**.
   - Ela copia os arquivos de imagens e anotações corretamente pareados para seus respectivos diretórios, garantindo que o dataset esteja pronto para o treino imediato.

6. **Geração do Arquivo de Configuração YAML**:
   - A função gera um **arquivo YAML** necessário pelo YOLO para especificar o número de classes, nomes das classes e os caminhos para os datasets de treino e validação.
   - Esse arquivo YAML é salvo na raiz do diretório do dataset YOLO, facilitando o início do treino com configuração mínima adicional.

---

### Fluxo de Trabalho:

1. **Carregar Arquivos de Imagem**:
   - A função começa buscando imagens no diretório fornecido, filtrando pelos formatos de imagem suportados (por exemplo, `.jpeg`, `.png`).
2. **Extrair Quadros de Vídeos** (se vídeos forem fornecidos):

   - Para cada vídeo, quadros são extraídos na taxa de quadros especificada. Cada quadro é salvo como uma imagem individual com um nome de arquivo único.

3. **Carregar Anotações**:

   - Os arquivos de anotações correspondentes `.txt` são carregados do diretório de anotações.

4. **Parear Imagens/Quadros com Anotações**:

   - A função verifica a consistência entre os pares de imagem/anotação e remove quaisquer arquivos não correspondentes. Ela garante que apenas imagens com anotações correspondentes sejam incluídas no dataset.

5. **Divisão entre Treino e Validação**:

   - O dataset é dividido em conjuntos de treino e validação, garantindo que a distribuição das classes permaneça equilibrada usando uma técnica de amostragem estratificada.

6. **Criar Estrutura de Pastas para YOLOv8**:

   - A estrutura de diretórios necessária para o YOLOv8 (`train/images`, `train/labels`, `valid/images`, `valid/labels`) é criada, e os arquivos de imagem/anotação são copiados para seus respectivos diretórios.

7. **Gerar Arquivo de Configuração YAML**:

   - Um arquivo YAML é gerado automaticamente com informações sobre o dataset (por exemplo, nomes das classes, número de classes e caminhos para os dados de treino/validação).

8. **Limpeza**:
   - Após o processamento, a pasta temporária usada para armazenar os quadros de vídeo extraídos é excluída para economizar espaço e manter a organização.

---

### Exemplos de Uso:

1. **Pré-processamento de um Dataset com Imagens e Vídeos**:

   ```python
   preprocess_dataset(
       imgs_path="caminho/para/imagens",
       txts_path="caminho/para/anotacoes",
       videos_path="caminho/para/videos",
       classes_txt="caminho/para/classes.txt",
       yolo_ds_path="caminho/para/dataset_yolo",
       val_size=0.2,  # 20% de divisão para validação
       frame_rate=1   # Extrair 1 quadro por segundo dos vídeos
   )
   ```

2. **Pré-processamento de um Dataset com Apenas Vídeos**:

   ```python
   preprocess_dataset(
       videos_path="caminho/para/videos",
       txts_path="caminho/para/anotacoes",
       classes_txt="caminho/para/classes.txt",
       yolo_ds_path="caminho/para/dataset_yolo",
       val_size=0.2,  # 20% de divisão para validação
       frame_rate=2   # Extrair 2 quadros por segundo dos vídeos
   )
   ```

3. **Pré-processamento de um Dataset com Apenas Imagens**:
   ```python
   preprocess_dataset(
       imgs_path="caminho/para/imagens",
       txts_path="caminho/para/anotacoes",
       classes_txt="caminho/para/classes.txt",
       yolo_ds_path="caminho/para/dataset_yolo",
       val_size=0.2  # 20% de divisão para validação
   )
   ```

---

### Conclusão:

Essa função simplifica muito o pipeline de pré-processamento para o treino com YOLO, facilitando a preparação de datasets a partir de imagens estáticas e vídeos. Automatizando etapas importantes como extração de quadros, divisão entre treino e validação e estruturação de arquivos, a função garante um dataset consistente e pronto para uso em tarefas de detecção de objetos.

Esta melhoria será particularmente útil a gente que quer trabalhar com datasets de vídeo, pois agiliza todo o fluxo de trabalho de extração de quadros e preparação de dados para o YOLO

```

```
