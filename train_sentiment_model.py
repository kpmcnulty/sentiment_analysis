import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, shuffle_on_init=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Store data together and optionally shuffle
        self.data = list(zip(texts, labels))
        if shuffle_on_init:
            import random
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_training_data(training_dir="training_data"):
    """Load all training data files from the training directory"""
    all_data = []
    
    # Get all training data files
    training_files = [f for f in os.listdir(training_dir) if f.startswith("training_data_") and f.endswith(".json")]
    
    for file in training_files:
        file_path = os.path.join(training_dir, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue
    
    return all_data

def prepare_data_for_training(training_data):
    """Convert training data into format suitable for model training"""
    texts = []
    labels = []
    
    # Rating mapping (1-5 to 0-4 for model labels)
    rating_to_label = {
        1: 0,  # very negative
        2: 1,  # negative
        3: 2,  # neutral
        4: 3,  # positive
        5: 4   # very positive
    }
    
    for item in training_data:
        texts.append(item['chunk_text'])
        labels.append(rating_to_label[item['rating']])
    
    return texts, labels

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def train_model(model_name="nlptown/bert-base-multilingual-uncased-sentiment", 
                training_dir="training_data",
                output_dir="models",
                num_epochs=4,  # Increased but still conservative
                batch_size=1,  # Individual examples for better calibration
                learning_rate=5e-6,  # 5x higher but still conservative
                shuffle_each_epoch=True,  # Reshuffle data each epoch
                continue_training=False):
    """Train the sentiment analysis model"""
    
    # Create output directorys
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(output_dir, f"sentiment_model_{timestamp}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading training data...")
    training_data = load_training_data(training_dir)
    
    if not training_data:
        raise ValueError("No training data found!")
    
    print(f"Loaded {len(training_data)} training examples")
    
    # Prepare data with optional weighting for recent corrections
    texts, labels = prepare_data_for_training(training_data)
    
    # Log class distribution
    label_counts = {i: labels.count(i) for i in range(5)}
    print(f"Label distribution: {label_counts}")
    
    # Split into train and validation sets with stratification
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, 
        test_size=0.15,  # Smaller validation split to have more training data
        random_state=42,
        stratify=labels  # Ensure balanced split
    )
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(val_texts)}")
    
    # Initialize tokenizer and model
    print("Initializing model and tokenizer...")
    
    # Initialize latest_model
    latest_model = None
    
    if continue_training:
        # Find the most recent model
        model_dirs = [d for d in os.listdir(output_dir) if d.startswith("sentiment_model_")]
        if not model_dirs:
            print("No previous model found, starting from base model")
            continue_training = False
        else:
            latest_model = sorted(model_dirs)[-1]
            model_path = os.path.join(output_dir, latest_model)
            # Convert to absolute path for transformers
            model_path = os.path.abspath(model_path)
            print(f"Continuing training from: {model_path}")
            
            # Check if model files exist
            config_file = os.path.join(model_path, "config.json")
            model_file = os.path.join(model_path, "pytorch_model.bin")
            
            if not os.path.exists(config_file):
                print(f"WARNING: config.json not found in {model_path}")
                print("Cannot continue from this model, starting fresh")
                continue_training = False
                latest_model = None
            elif not os.path.exists(model_file):
                print(f"WARNING: pytorch_model.bin not found in {model_path}")
                print("Cannot continue from this model, starting fresh")
                continue_training = False
                latest_model = None
            else:
                try:
                    # Check if config has model_type
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        if 'model_type' not in config:
                            print(f"WARNING: model_type missing from config.json")
                            print("This model appears corrupted, starting fresh")
                            continue_training = False
                            latest_model = None
                        else:
                            tokenizer = AutoTokenizer.from_pretrained(model_path)
                            model = AutoModelForSequenceClassification.from_pretrained(model_path)
                except Exception as e:
                    print(f"Error loading previous model: {str(e)}")
                    print("Falling back to base model")
                    continue_training = False
                    latest_model = None
    
    # Load base model if not continuing or if continue failed
    if not continue_training:
        print("Starting fresh training from base model")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5  # 5 sentiment classes
        )
    
    # Create datasets with shuffling
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, shuffle_on_init=True)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, shuffle_on_init=False)
    
    # Ensure the model config is preserved before training
    # This prevents issues with checkpointing
    initial_config_path = os.path.join(model_output_dir, "config.json")
    model.config.save_pretrained(model_output_dir)
    
    # Training arguments with conservative settings for small dataset
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,  # Can use larger batch for evaluation
        warmup_ratio=0.1,  # 10% of training steps for warmup
        weight_decay=0.2,  # Stronger regularization
        logging_dir=os.path.join(model_output_dir, 'logs'),
        logging_steps=5,  # More frequent logging
        learning_rate=learning_rate,
        eval_strategy="steps",  # Evaluate more frequently
        eval_steps=25,  # Evaluate every 25 steps
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=True,  # Load best model at end of training
        metric_for_best_model="eval_f1",  # Use F1 score instead of loss
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        gradient_accumulation_steps=8,  # With batch_size=1, this gives effective batch of 8
        max_grad_norm=0.5,  # Gradient clipping for stability
        dataloader_drop_last=False,  # Don't drop last batch with small dataset
        report_to="none"  # Disable wandb/tensorboard for simplicity
    )
    
    # Initialize trainer with metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # Stop if validation metric doesn't improve
    )
    
    # Evaluate baseline performance before training
    print("Evaluating baseline performance...")
    baseline_metrics = trainer.evaluate()
    print(f"Baseline metrics: Accuracy={baseline_metrics.get('eval_accuracy', 0):.3f}, F1={baseline_metrics.get('eval_f1', 0):.3f}")
    
    # Save baseline metrics
    baseline_results = {
        'baseline_accuracy': baseline_metrics.get('eval_accuracy', 0),
        'baseline_f1': baseline_metrics.get('eval_f1', 0),
        'baseline_loss': baseline_metrics.get('eval_loss', 0)
    }
    
    # Train the model
    print("Starting training...")
    train_result = trainer.train()
    
    # Get final metrics
    final_metrics = trainer.evaluate()
    print(f"Final metrics: Accuracy={final_metrics.get('eval_accuracy', 0):.3f}, F1={final_metrics.get('eval_f1', 0):.3f}")
    
    # Calculate improvement
    accuracy_improvement = final_metrics.get('eval_accuracy', 0) - baseline_metrics.get('eval_accuracy', 0)
    f1_improvement = final_metrics.get('eval_f1', 0) - baseline_metrics.get('eval_f1', 0)
    print(f"Improvement: Accuracy={accuracy_improvement:+.3f}, F1={f1_improvement:+.3f}")
    
    # Save the model and tokenizer with all config
    print("Saving model...")
    # Save with safe_serialization=False to ensure compatibility
    model.save_pretrained(model_output_dir, safe_serialization=False)
    tokenizer.save_pretrained(model_output_dir)
    
    # Verify config.json was created with model_type
    config_path = os.path.join(model_output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            if 'model_type' not in saved_config:
                print("WARNING: model_type missing from config.json")
                # Add model_type from the original model
                saved_config['model_type'] = model.config.model_type
                with open(config_path, 'w') as f:
                    json.dump(saved_config, f, indent=2)
    
    # Save training configuration with metrics
    config = {
        "model_name": model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_examples": len(training_data),
        "timestamp": timestamp,
        "continued_from": latest_model if continue_training else None,
        "train_set_size": len(train_texts),
        "val_set_size": len(val_texts),
        "label_distribution": label_counts,
        "baseline_metrics": baseline_results,
        "final_metrics": {
            'accuracy': final_metrics.get('eval_accuracy', 0),
            'f1': final_metrics.get('eval_f1', 0),
            'loss': final_metrics.get('eval_loss', 0)
        },
        "improvement": {
            'accuracy': accuracy_improvement,
            'f1': f1_improvement
        },
        "training_steps": train_result.global_step
    }
    
    with open(os.path.join(model_output_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save training history for plotting
    history = trainer.state.log_history
    with open(os.path.join(model_output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create a simple training curve plot
    try:
        import matplotlib.pyplot as plt
        
        # Extract metrics from history
        train_losses = [h.get('loss', None) for h in history if 'loss' in h]
        eval_losses = [h.get('eval_loss', None) for h in history if 'eval_loss' in h]
        eval_f1s = [h.get('eval_f1', None) for h in history if 'eval_f1' in h]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        if train_losses:
            ax1.plot(train_losses, label='Train Loss', marker='o')
        if eval_losses:
            ax1.plot([i for i in range(len(eval_losses))], eval_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress - Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot F1 scores
        if eval_f1s:
            ax2.plot([i for i in range(len(eval_f1s))], eval_f1s, label='Val F1', marker='s', color='green')
        ax2.set_xlabel('Evaluation Steps')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Training Progress - F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, "training_curves.png"))
        plt.close()
        print(f"Training curves saved to: {os.path.join(model_output_dir, 'training_curves.png')}")
    except Exception as e:
        print(f"Could not create training curves: {str(e)}")
    
    print(f"Training complete! Model saved to: {model_output_dir}")
    return model_output_dir

if __name__ == "__main__":
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Ask user if they want to continue training
        continue_training = input("Continue training from previous model? (y/n): ").lower() == 'y'
        
        model_dir = train_model(continue_training=continue_training)
        print("\nTraining completed successfully!")
        print(f"Model saved to: {model_dir}")
        print("\nTo use this model, update your sentiment_config.json with:")
        print(f'"sentiment_model": "{model_dir}"')
    except Exception as e:
        print(f"Error during training: {str(e)}") 