import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
from models.soft_vqvae import SoftVQVAE
from pixelcnn.models import GatedPixelCNN
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def extract_latent_codes(model, data_loader, device):
    """
    Extract latent codes from VQVAE model
    """
    model.eval()
    all_codes = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(data_loader):
            x = x.to(device)
            
            # Get latent codes from VQVAE - both models use encoder attribute
            z = model.encoder(x)
            z = model.pre_quantization_conv(z)
            
            # Get quantized codes and indices
            if hasattr(model, 'vector_quantization'):
                # For VQVAE
                embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = model.vector_quantization(z)
                
                # Use the indices directly from the vector quantizer
                indices = min_encoding_indices.squeeze(1)
                
            else:
                # For SoftVQVAE
                # 1. 正常获取高质量软向量 z_q
                z_q, attn_weights = model.quantizer(z) 

                # 2. 现在，对 z_q 进行离散化，而不是对 z
                z_q_flat = z_q.permute(0, 2, 3, 1).contiguous()
                z_q_flat = z_q_flat.view(-1, model.quantizer.embedding_dim)

                # 3. 计算 z_q 和码本之间的距离
                distances = (z_q_flat.unsqueeze(1) - model.quantizer.codebook.weight.unsqueeze(0)).pow(2).sum(2)

                # 4. 找到最近的索引
                indices = torch.argmin(distances, dim=1)

            
            # Reshape indices to match latent spatial dimensions
            latent_shape = z.shape[2:]
            indices = indices.view(x.shape[0], *latent_shape)
            
            all_codes.append(indices.cpu())
            all_labels.append(labels)
    
    all_codes = torch.cat(all_codes, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_codes, all_labels


def train_pixelcnn(model, model_name, training_loader, validation_loader, args, device):
    """
    Train PixelCNN model on latent codes
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    results = {
        'model_name': model_name,
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }
    
    model.train()
    
    for epoch in range(args.epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (codes, labels) in enumerate(training_loader):
            codes = codes.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(codes, labels)
            
            # Reshape for loss calculation
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, args.n_embeddings),
                codes.view(-1)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == codes).sum().item()
            train_total += codes.numel()
            
            if batch_idx % args.log_interval == 0:
                accuracy = train_correct / train_total if train_total > 0 else 0
                print(f'{model_name} - Epoch {epoch}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (codes, labels) in enumerate(validation_loader):
                if batch_idx >= 10:  # Use first 10 batches for validation
                    break
                    
                codes = codes.to(device)
                labels = labels.to(device)
                
                logits = model(codes, labels)
                logits = logits.permute(0, 2, 3, 1).contiguous()
                
                loss = criterion(
                    logits.view(-1, args.n_embeddings),
                    codes.view(-1)
                )
                
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == codes).sum().item()
                val_total += codes.numel()
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(training_loader)
        avg_val_loss = val_loss / min(10, len(validation_loader))
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        results['train_losses'].append(avg_train_loss)
        results['val_losses'].append(avg_val_loss)
        results['train_accuracies'].append(train_accuracy)
        results['val_accuracies'].append(val_accuracy)
        
        print(f'{model_name} - Epoch {epoch} Summary:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        model.train()
    
    return results


def generate_samples_from_pixelcnn(pixelcnn_model, vqvae_model, labels, shape, device):
    """
    Generate samples using PixelCNN and decode with VQVAE
    """
    pixelcnn_model.eval()
    vqvae_model.eval()
    
    with torch.no_grad():
        # Generate latent codes using PixelCNN
        generated_codes = pixelcnn_model.generate(
            labels, shape=shape, batch_size=len(labels)
        )
        
        # Convert codes to embeddings
        if hasattr(vqvae_model, 'vector_quantization'):
            # VQVAE
            generated_codes_flat = generated_codes.view(-1)
            embeddings = vqvae_model.vector_quantization.embedding(generated_codes_flat)
            embeddings = embeddings.view(generated_codes.shape[0], -1, vqvae_model.vector_quantization.e_dim)
            embeddings = embeddings.permute(0, 2, 1).contiguous()
            embeddings = embeddings.view(generated_codes.shape[0], vqvae_model.vector_quantization.e_dim, *shape)
        else:
            # SoftVQVAE
            generated_codes_flat = generated_codes.view(-1)
            embeddings = vqvae_model.quantizer.codebook(generated_codes_flat)
            embeddings = embeddings.view(generated_codes.shape[0], -1, vqvae_model.quantizer.embedding_dim)
            embeddings = embeddings.permute(0, 2, 1).contiguous()
            embeddings = embeddings.view(generated_codes.shape[0], vqvae_model.quantizer.embedding_dim, *shape)
        
        # Decode to images
        generated_images = vqvae_model.decoder(embeddings)
    
    return generated_images


def evaluate_pixelcnn_quality(pixelcnn_model, vqvae_model, test_loader, device):
    """
    Evaluate the quality of generated samples
    """
    pixelcnn_model.eval()
    vqvae_model.eval()
    
    real_images = []
    generated_images = []
    
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_loader):
            if batch_idx >= 5:  # Use first 5 batches for evaluation
                break
            
            x = x.to(device)
            labels = labels.to(device)
            
            # Get latent shape from VQVAE
            if hasattr(vqvae_model, 'encode'):
                # VQVAE
                z = vqvae_model.encode(x)
                latent_shape = z.shape[2:]
            else:
                # SoftVQVAE
                z = vqvae_model.encoder(x)
                latent_shape = z.shape[2:]
            
            # Generate samples
            gen_imgs = generate_samples_from_pixelcnn(
                pixelcnn_model, vqvae_model, labels, latent_shape, device
            )
            
            real_images.append(x.cpu())
            generated_images.append(gen_imgs.cpu())
    
    # Calculate metrics (simplified - you can add more sophisticated metrics)
    real_images = torch.cat(real_images, dim=0)
    generated_images = torch.cat(generated_images, dim=0)
    
    # MSE between real and generated
    mse = torch.mean((real_images - generated_images) ** 2).item()
    
    return {
        'mse': mse,
        'real_images': real_images,
        'generated_images': generated_images
    }


def plot_pixelcnn_results(results_dict, save_dir):
    """
    Plot PixelCNN training results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 3, 1)
    for model_name, results in results_dict.items():
        plt.plot(results['train_losses'], label=f'{model_name} Train')
        plt.plot(results['val_losses'], label=f'{model_name} Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PixelCNN Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training and validation accuracy
    plt.subplot(2, 3, 2)
    for model_name, results in results_dict.items():
        plt.plot(results['train_accuracies'], label=f'{model_name} Train')
        plt.plot(results['val_accuracies'], label=f'{model_name} Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('PixelCNN Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot comparison of final validation losses
    plt.subplot(2, 3, 3)
    final_val_losses = [results['val_losses'][-1] for results in results_dict.values()]
    plt.bar(results_dict.keys(), final_val_losses)
    plt.ylabel('Final Validation Loss')
    plt.title('Final Validation Loss Comparison')
    plt.grid(True)
    
    # Plot comparison of final validation accuracies
    plt.subplot(2, 3, 4)
    final_val_accuracies = [results['val_accuracies'][-1] for results in results_dict.values()]
    plt.bar(results_dict.keys(), final_val_accuracies)
    plt.ylabel('Final Validation Accuracy')
    plt.title('Final Validation Accuracy Comparison')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pixelcnn_training_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PixelCNN Ablation Study for VQVAE Latent Spaces')
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--dataset", type=str, default='CIFAR10')
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--img_dim", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature parameter for SoftVQVAE")
    
    # VQVAE model paths
    parser.add_argument("--vqvae_model_path", type=str, default=None)
    parser.add_argument("--softvqvae_model_path", type=str, default=None)
    
    # Experiment settings
    parser.add_argument("--save_dir", type=str, default='pixelcnn_ablation_results')
    parser.add_argument("--save_samples", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save experiment configuration
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size)
    
    # Load pre-trained VQVAE models
    vqvae_models = {}
    
    if args.vqvae_model_path and os.path.exists(args.vqvae_model_path):
        # Load VQVAE
        vqvae = VQVAE(128, 32, 2, args.n_embeddings, 64, 0.25).to(device)
        vqvae.load_state_dict(torch.load(args.vqvae_model_path, map_location=device))
        vqvae_models['VQVAE'] = vqvae
        print("Loaded VQVAE model")
    
    if args.softvqvae_model_path and os.path.exists(args.softvqvae_model_path):
        # Load SoftVQVAE
        softvqvae = SoftVQVAE(
            h_dim=128, res_h_dim=32, n_res_layers=2,
            num_embeddings=args.n_embeddings, embedding_dim=64, beta=0.25, temperature=args.temperature
        ).to(device)
        softvqvae.load_state_dict(torch.load(args.softvqvae_model_path, map_location=device))
        vqvae_models['SoftVQVAE'] = softvqvae
        print("Loaded SoftVQVAE model")
    
    if not vqvae_models:
        # If no pre-trained models provided, use default models
        vqvae_models['VQVAE'] = VQVAE(128, 32, 2, args.n_embeddings, 64, 0.25).to(device)
        vqvae_models['SoftVQVAE'] = SoftVQVAE(
            h_dim=128, res_h_dim=32, n_res_layers=2,
            num_embeddings=args.n_embeddings, embedding_dim=64, beta=0.25, temperature=args.temperature
        ).to(device)
        print("Using default VQVAE models")
    
    # Extract latent codes from each VQVAE
    print("Extracting latent codes from VQVAE models...")
    latent_datasets = {}
    
    for model_name, model in vqvae_models.items():
        codes, labels = extract_latent_codes(model, training_loader, device)
        latent_datasets[model_name] = (codes, labels)
        print(f"{model_name}: Extracted {len(codes)} latent codes with shape {codes.shape}")
    
    # Create PixelCNN models for each latent space
    pixelcnn_models = {}
    
    for model_name, (codes, labels) in latent_datasets.items():
        # Get latent shape
        latent_shape = codes.shape[1:]  # (H, W)
        
        # Create PixelCNN model
        pixelcnn = GatedPixelCNN(
            input_dim=args.n_embeddings,
            dim=args.img_dim**2,
            n_layers=args.n_layers,
            n_classes=10  # Assuming 10 classes for CIFAR10
        ).to(device)
        
        pixelcnn_models[model_name] = pixelcnn
    
    # Create data loaders for latent codes
    from torch.utils.data import TensorDataset, DataLoader
    
    latent_loaders = {}
    for model_name, (codes, labels) in latent_datasets.items():
        dataset = TensorDataset(codes, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        latent_loaders[model_name] = (train_loader, val_loader)
    
    # Train PixelCNN models
    print("\nStarting PixelCNN training...")
    results_dict = {}
    
    for model_name, pixelcnn in pixelcnn_models.items():
        print(f"\n=== Training PixelCNN on {model_name} latent space ===")
        
        train_loader, val_loader = latent_loaders[model_name]
        
        results = train_pixelcnn(
            pixelcnn, f"PixelCNN-{model_name}", train_loader, val_loader, args, device
        )
        
        results_dict[model_name] = results
        
        # Save trained PixelCNN model
        model_path = os.path.join(save_dir, f'pixelcnn_{model_name.lower()}.pth')
        torch.save(pixelcnn.state_dict(), model_path)
        print(f"PixelCNN model saved to {model_path}")
    
    # Evaluate and compare
    print("\n=== Evaluating PixelCNN models ===")
    evaluation_results = {}
    
    for model_name, pixelcnn in pixelcnn_models.items():
        print(f"Evaluating {model_name}...")
        
        vqvae_model = vqvae_models[model_name]
        eval_results = evaluate_pixelcnn_quality(pixelcnn, vqvae_model, validation_loader, device)
        evaluation_results[model_name] = eval_results
        
        print(f"  MSE: {eval_results['mse']:.6f}")
        
        # Save generated samples
        if args.save_samples:
            sample_path = os.path.join(save_dir, f'generated_samples_{model_name.lower()}.png')
            save_image(eval_results['generated_images'][:16], sample_path, nrow=4, normalize=True)
            print(f"Generated samples saved to {sample_path}")
    
    # Save results
    with open(os.path.join(save_dir, 'pixelcnn_training_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    with open(os.path.join(save_dir, 'pixelcnn_evaluation_results.json'), 'w') as f:
        # Convert tensors to lists for JSON serialization
        serializable_eval_results = {}
        for model_name, results in evaluation_results.items():
            serializable_eval_results[model_name] = {
                'mse': results['mse'],
                'real_images_shape': list(results['real_images'].shape) if results['real_images'] is not None else None,
                'generated_images_shape': list(results['generated_images'].shape) if results['generated_images'] is not None else None
            }
        json.dump(serializable_eval_results, f, indent=2)
    
    # Generate comparison plots
    print("Generating comparison plots...")
    plot_pixelcnn_results(results_dict, save_dir)
    
    # Print final comparison
    print("\n=== Final PixelCNN Comparison ===")
    for model_name, results in results_dict.items():
        final_val_loss = results['val_losses'][-1]
        final_val_accuracy = results['val_accuracies'][-1]
        print(f"{model_name}:")
        print(f"  Final Validation Loss: {final_val_loss:.4f}")
        print(f"  Final Validation Accuracy: {final_val_accuracy:.4f}")
    
    print(f"\nPixelCNN ablation experiment completed! Results saved to: {save_dir}")


if __name__ == "__main__":
    main()