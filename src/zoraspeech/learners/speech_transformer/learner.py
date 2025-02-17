from zoraspeech.architectures.speech_transformer.speech_transformer import SpeechTransformer, Config
import einops
import torch as t
from torch.utils.data import DataLoader, random_split
from zoraspeech.datasets.speech_transformer.speech_transformer_dataset import CommonVoiceDataset, CharacterVocabulary, create_collate_fn
from jaxtyping import Float, Int, Bool
from tqdm import tqdm
import wandb
import json
import csv
from datetime import datetime
from dataclasses import asdict
import os

class LearnerLogger:
    def __init__(self, csv_filename, headers):
        self.log = {}
        self.csv_filename = csv_filename
    
        # initialize csv and write header row
        #self.headers = headers # ['step', 'loss', 'learning_rate', 'gradient_norms'] 

        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    def save_csv(self, metric):
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(metric)

class SpeechTransformerLearner:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = SpeechTransformer(self.cfg).to(self.cfg.device)
        # loss function ??
        self.optimizer = t.optim.AdamW(
            self.model.parameters(),
            lr=1e-2, # how do we implement this variable learning rate? can we just try a fixed value for now?
            betas=(self.cfg.op_beta_1, self.cfg.op_beta_2),
            eps=self.cfg.op_eps
        )
        self.steps = 1
        self.use_wandb = True
        self.cv = CharacterVocabulary(cfg)
        self.best_wer = float('inf')
        self.best_loss = float('inf')

    def compute_loss(
              self,
              probabilities: Float[t.Tensor, "batch seq_length d_vocab"], #type: ignore
              tokens: Int[t.Tensor, "batch seq_length"], #type: ignore
              padding_mask: Bool[t.Tensor, "batch seq_length"] #type: ignore
              ) -> Float[t.Tensor, "batch seq_length-1"]: #type: ignore
        """
        Usage: loss = -calculate_log_probs(logits, inputs).mean()
        GPT-2 Implementation for reference
        log_probs = logits.log_softmax(dim=-1)
        log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

        Input: logits, encoded text tokens
        Output: negative log probabilities of the first seq_length - 1 predictions (so we can compare them with the actual next tokens)
        
        Compute cross entropy loss with:
        - Label smoothing (0.8 for correct token, 0.2/(V-1) for others)
        - Padding mask to ignore pad tokens
        - Shifted prediction (predict next token)
        """
        # handle teacher forcing alignment
        probabilities = probabilities[:, :-1, :]
        tokens = tokens[:, 1:]
        padding_mask = padding_mask[:, 1:]

        #print("After alignment:")
        #print(f"probabilities: {probabilities.shape}")
        #print(f"tokens: {tokens.shape}")
        #print(f"padding_mask: {padding_mask.shape}")

        # apply label smoothed target distribution
        smoothed_target_distribution = t.zeros(probabilities.shape).to(probabilities.device)
        smoothed_target_distribution.fill_(0.2 / (self.cfg.vocab_size - 1))
        indices = einops.rearrange(tokens, "b s -> b s 1")
        smoothed_target_distribution.scatter_(-1, indices, 0.8)

        assert (smoothed_target_distribution.sum(dim=-1) - 1.0).abs().max() < 1e-6

        # compute cross entry with smoothed targets
        log_probs = t.log(probabilities + 1e-10) # small eps to prevent log(0)
        log_probs = t.mul(log_probs, smoothed_target_distribution)
        sum = log_probs.sum(dim=-1)
        masked_sum = sum.masked_fill_(~padding_mask, 0) # invert mask so we zero out padded tokens (which come in as False)
        loss = masked_sum.mean()        

        return -loss

    def get_learning_rate(self, steps):
        # lrate = k · d−0.5  model · min(n−0.5, n · warmup n−1.5), from paper
        #TODO modify this later to use a variable k value once the model converges
        #print("Steps: ", steps)
        lrate = self.cfg.k_fixed * (self.cfg.d_model ** -0.5) * min( (steps ** -0.5), (steps * (self.cfg.warmup_n ** -1.5) ) )
        return lrate

    def learning_step(self, audio_feature, text, padding_mask):
        
        #audio_features, text, padding_mask = batch['audio_features'].to(self.cfg.device), batch['text'].to(self.cfg.device), batch['padding_mask'].to(self.cfg.device)
        probabilities = self.model(audio_feature, text)
        loss = self.compute_loss(probabilities, text, padding_mask)


        loss.backward()
        
        # monitor gradients
        total_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2) # L2 norm
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        
        # clip gradients in case they have exploded
        t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip_value) # gradient clipping
        
        # update learning rate
        lr = self.get_learning_rate(self.steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, lr, total_norm

    def learn(self):
        # training loop happens here

        try:
            # set up dataset
            full_dataset_tsv_path = "/data/jo/commonvoice/cv-corpus-19.0-2024-09-13/en/validated.tsv"
            clips_path = "/data/jo/commonvoice/cv-corpus-19.0-2024-09-13/en/clips_wav"
            full_dataset = CommonVoiceDataset(self.cfg, full_dataset_tsv_path, clips_path)  

            # set up dataloader
            train_size = int(0.7 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            print('train_size:', train_size, 'val_size', val_size)
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=self.cfg.batch_size, 
                shuffle=True, 
                collate_fn=create_collate_fn(self.cv, self.cfg)
                )
            
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=self.cfg.batch_size, 
                shuffle=True, 
                collate_fn=create_collate_fn(self.cv, self.cfg)
                )

            # set up loggers
            training_logger = LearnerLogger(
                csv_filename='training_logger.csv',
                headers=['step', 'loss', 'wer', 'learning_rate', 'gradient_norms']
                )

            # set up wandb
            with wandb.init(
                project="zora_speech_transformer",
                name = "zora_speech_transformer_" + str(datetime.now()),
                config=asdict(self.cfg)
            ) as run:

                while self.steps < self.cfg.training_steps:

                    progress_bar = tqdm(total = len(train_dataloader), desc="Learning")

                    for batch in train_dataloader:
                        for i in range(len(batch['all_audio_features'])):
                            audio_feature = batch['all_audio_features'][i].to(self.cfg.device)
                            text = batch['all_texts'][i].to(self.cfg.device)
                            padding_mask = batch['all_text_masks'][i].to(self.cfg.device)

                            #print("audio feature shape: ", audio_feature.shape)
                            #print("audio feature: ", audio_feature, "text: ", text, "padding mask: ", padding_mask)
                            #print('string: ', self.cv.decode(text[0].tolist()))
                            
                            loss, lr, grad_norm = self.learning_step(audio_feature, text, padding_mask)

                            progress_bar.update()
                            progress_bar.set_description(f'Steps {self.steps+1}/{self.cfg.training_steps}, Loss = {loss:.2f}, LR = {lr:.2f}, GN = {grad_norm:.2f}')

                            # log 
                            wandb.log(
                                {
                                'steps': self.steps,
                                'loss': loss,
                                'lr': lr,
                                'grad_norm': grad_norm,
                                }
                                )
                            training_logger.save_csv([self.steps, loss, '-', self.get_learning_rate(self.steps), grad_norm])

                            if self.steps % self.cfg.validation_frequency == 0:
                                metrics = self.run_validation(val_dataloader)
                                # log metrics 
                                print(metrics)
                                wandb.log(
                                    {
                                    'steps': self.steps,
                                    'loss': metrics['loss'],
                                    'wer': metrics['wer'],
                                    }
                                    )

                                training_logger.save_csv([self.steps, metrics['loss'], metrics['wer'], self.get_learning_rate(self.steps), 'fill_with_gradient_norm'])

                                # update progress bar
                                progress_bar.set_description(f'Steps {self.steps+1}/{self.cfg.training_steps}, Loss = {metrics['loss']:.2f} WER = {metrics['wer']:.2f}')
                            
                            self.steps += 1
                            
                            if self.steps >= self.cfg.training_steps:
                                # finished training
                                progress_bar.close()
                                break 
                        if self.steps >= self.cfg.training_steps:
                            # finished training
                            progress_bar.close()
                            break 
                    if self.steps >= self.cfg.training_steps:
                        # finished training
                        progress_bar.close()
                        break 


                
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            raise # re-reaise the exception after cleanup
        finally:
                if hasattr(self, 'model'):
                    self.model.to('cpu')
        t.cuda.empty_cache()
        try:
            wandb.finish()
        except:
            pass

    def run_validation(self, val_dataloader):
        print("running validation")
        # Set model to eval mode
        self.model.eval()
        # Run validation loop
        metrics = []

        with t.no_grad():
            progress_bar = tqdm(total = len(val_dataloader), desc="Validation")
            for batch in val_dataloader:

                for i in range(len(batch['all_audio_features'])):
                    audio_feature = batch['all_audio_features'][i].to(self.cfg.device)
                    text = batch['all_texts'][i].to(self.cfg.device)
                    padding_mask = batch['all_text_masks'][i].to(self.cfg.device)


                    batch_metrics =self.validation_step(audio_feature, text, padding_mask) 
                    metrics.append(batch_metrics)
                    progress_bar.update()
                    progress_bar.set_description(f'Current WER = {batch_metrics['wer']:.2f}, Loss = {batch_metrics['loss']:.2f}')

        average_metrics = {
            'wer': sum(m['wer'] for m in metrics) / len(metrics),
            'loss': sum(m['loss'] for m in metrics) / len(metrics),
        }
        progress_bar.set_description(f'WER = {average_metrics['wer']:.2f}')

        # turn model back to training mode
        self.model.train()

        # close progress bar
        progress_bar.close()

        # save checkpoint
        self.save_checkpoint(average_metrics)

        #print(average_metrics)

        return average_metrics


    def validation_step(self, audio_feature, text, padding_mask):
        
        #audio_features, text, padding_mask = batch['audio_features'].to(self.cfg.device), batch['text'].to(self.cfg.device), batch['padding_mask'].to(self.cfg.device)
        probabilities = self.model(audio_feature, text)
        loss = self.compute_loss(probabilities, text, padding_mask)
        
        wer = self.get_wer(probabilities, text)
        #cer = self.get_cer()

        if loss < self.best_loss:
            self.best_loss = loss

        if wer < self.best_wer:
            self.best_wer = wer

        return {
            'probabilities': probabilities,
            'loss': loss,
            'wer': wer
            #'cer': cer
        }

    def get_wer(self, probabilities, text):
        # WER = (Substitutions + Deletions + Insertions) / (Total Words in Ground Truth)
        
        predicted_characters = t.argmax(probabilities, dim=-1)[0].cpu().tolist()
        predicted_text = self.cv.decode(predicted_characters)
        
        actual_text = self.cv.decode(text[0].cpu().tolist())

        #print(f"Predicted text: {predicted_text}")
        #print(f"Actual text: {actual_text}")
        
        if actual_text and predicted_text == "":
            return 1.0

        predicted_words = predicted_text.split(' ')
        actual_words = actual_text.split(' ')

        # get number of edits with Levenshtein distance algorithm

        # create matrix with extra row/column for empty string case
        rows = len(predicted_words) + 1
        cols = len(actual_words) + 1
        dp = [[0 for _ in range(cols)] for _ in range(rows) ]

        # inittialize first row and column
        for r in range(rows):
            dp[r][0] = r # cost of deleting r words
        for c in range(cols):
            dp[0][c] = c # cost of inserting c words

        # fill the matrix
        for r in range(1, rows):
            for c in range(1, cols):
                if predicted_words[r - 1] == actual_words[c - 1]:
                    substitution_cost = 0
                else:
                    substitution_cost = 1

                dp[r][c] = min(
                    dp[r-1][c] + 1,
                    dp[r][c-1] + 1,
                    dp[r-1][c-1] + substitution_cost
                )
        # WER is the final cell divided by the length of actual words
        return dp[rows - 1][cols - 1] / len(actual_words)


    def get_cer(self):
        pass

    def save_checkpoint(self, metrics):
        checkpoints_dir ='/data/jo/checkpoints' 
        os.makedirs(checkpoints_dir, exist_ok=True)

        datetime_now_str = str(datetime.now())
        output_file_checkpoint = os.path.join(checkpoints_dir, "checkpoint_" + datetime_now_str + '.pth')
        #output_file_model = os.path.join(checkpoints_dir, "model_state_dict_" + datetime_now_str + '.pth')
        #output_file_optimizer = os.path.join(checkpoints_dir, "optimizer_state_dict_" + datetime_now_str + '.pt')


        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'steps': self.steps,
            'metrics': {
                'average_wer': metrics['wer'],
                'average_loss': metrics['loss'],
                'best_wer': self.best_wer,
                'best_loss': self.best_loss
            },
            'config': asdict(self.cfg)
        }

        t.save(checkpoint, output_file_checkpoint)

    def load_checkpoint(self, path):
        try:
            checkpoint = t.load(path)

            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.cfg = checkpoint['config']
            self.steps = checkpoint['steps']
            self.best_wer = checkpoint['metrics']['best_wer']
            self.best_loss = checkpoint['metrics']['best_loss']

        except RuntimeError:
            print("issue loading checkpoint")