import argparse
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data
from tqdm import tqdm
import os
import torch
import pickle as pkl
import pandas as pd

from configs.model import InputsConfig
from configs.paths import OUTPUT_ROOT, TEMP_DIR
from data.embeddings.dgi_new.model.DGIModule import DGIModule


def train(data, model, optimizer, scheduler, args):
    model.train()

    optimizer.zero_grad()

    # Forward pass
    pos_score, neg_score = model(data)
    # Cálculo da perda
    loss = model.loss(pos_score, neg_score)

    # Backpropagation
    loss.backward()

    # Clip de gradientes
    clip_grad_norm_(model.parameters(), max_norm=args.max_norm)

    # Atualização dos pesos
    optimizer.step()
    scheduler.step()

    return loss.item()


def create_embedding(output, inter_data, args):
    output_folder = f"{OUTPUT_ROOT}/{output}"
    os.makedirs(output_folder, exist_ok=True)

    # Instanciando o modelo
    model = DGIModule(hidden_channels=args.dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma, verbose=False)
    # scheduler = OneCycleLR(optimizer, max_lr=args.lr * 100,
    #                         steps_per_epoch=args.epoch, epochs=args.epoch)

    # Carregando os dados
    with open(inter_data, 'rb') as handle:
        city_dict = pkl.load(handle)

    data = Data(
        embedding_array=torch.tensor(city_dict['embedding_array'], dtype=torch.float32),
        x=torch.tensor(city_dict['embedding_array_test'], dtype=torch.float32),
        edge_index=torch.tensor(city_dict['edge_index'], dtype=torch.int64),
        edge_weight=torch.tensor(city_dict['edge_weight'], dtype=torch.float32),
        number_pois=city_dict['number_pois']
    )

    # Removendo a variável de rótulos y se necessário
    # data.y = None  # Não precisamos de rótulos para o aprendizado não supervisionado

    # Mover dados para o dispositivo correto
    place_ids = city_dict.get('place_id', [])
    category2 = city_dict.get('l_category', [])
    data = data.to(args.device)

    bar = tqdm(range(args.epoch), desc="Epochs de Treinamento")

    for epoch in bar:
        loss_train = train(data, model, optimizer, scheduler, args)
        bar.set_postfix(loss=loss_train, lr=optimizer.param_groups[0]['lr'])

    # Obter embeddings reais do encoder
    with torch.no_grad():
        model.eval()
        embeddings = model.poi_encoder(data.x, data.edge_index)

    # Salvando os embeddings
    output_path = f'{output_folder}/embeddings.csv'
    embeddings_np = embeddings.cpu().numpy()
    print(embeddings_np)
    print(place_ids)

    df = pd.DataFrame(embeddings_np, columns=[f'{i}' for i in range(embeddings_np.shape[1])])
    df.insert(0, 'placeid', place_ids)
    df.insert(1, 'category', category2)
    df.to_csv(output_path, index=False)
    print(f"Embeddings salvos em: {output_path}")
    print(f"Dimensões finais: {embeddings.shape} (deveria ser [num_pois, {args.dim * 4}])")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create embeddings for POI data')
    parser.add_argument('--city', type=str,
                        default='Florida',
                        help='City name for processing POI data')
    parser.add_argument('--output',
                        type=str,
                        default=parser.parse_args().city.lower() + "_dgi_new",
                        help='Output directory for processed data')
    parser.add_argument('--inter_data', type=str,
                        default=f"{TEMP_DIR}/dgi/{parser.parse_args().city}/data_inter.pkl",
                        help='Intermediate data file for processing POI data')

    parser.add_argument('--dim', type=int, default=InputsConfig.EMBEDDING_DIM,
                        help='Dimension of output representation')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--epoch', type=int, default=70)
    parser.add_argument('--device', type=str,
                        default='mps' if torch.backends.mps.is_available() else 'gpu' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_norm', type=float, default=0.9)

    args = parser.parse_args()

    create_embedding(output=args.output, inter_data=args.inter_data, args=args)
