import time
import clip
import numpy as np
from tqdm import tqdm
from core.utils import convert_models_to_fp32
import torch
import logging
from sklearn.metrics import confusion_matrix
from core.eval_utils import print_confusion_matrix


def train_class(args, model, weights_path, train_dataloader, test_dataloader, use_albumentations, params, optimizer, scheduler, loss_img, loss_txt, writer, device, pretrain=None):
    num_batches_train = len(train_dataloader.dataset)/args.batch_size
    num_batches_val = len(test_dataloader.dataset)/args.batch_size
    pred_list = []
    gt_list = []
    accuracy_list = []
    train_loss_list = []
    if pretrain:
        model.load_state_dict(torch.load(pretrain)['model_state_dict'])
    
    # train
    for epoch in range(args.num_epochs):
        epoch += 1
        print(f"Epoch: {epoch}")
        epoch_train_loss = 0
        model.train()

        for i, batch in enumerate(tqdm(train_dataloader,total=num_batches_train)):
            optimizer.zero_grad()

            images, class_ids, _  = batch
            if use_albumentations:
                images = images['image']

            images = torch.stack([img for img in images], dim=0).to(device)
            texts = [f"A photo of a {train_dataloader.dataset.classes[class_id]}." for class_id in class_ids]
            texts = clip.tokenize(texts).to(device)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(logits_per_image.shape[0], device=device)

            total_train_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_train_loss.backward()
            epoch_train_loss += total_train_loss

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            convert_models_to_fp32(model)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                writer.add_scalar("Log/LR_scheduler", torch.tensor(scheduler.get_last_lr()[0]), iteration_count)
            
            clip.model.convert_weights(model)

            iteration_count = epoch * len(train_dataloader) + i
            if i % args.print_log_by_iter == 0:
                print(f"epoch: {epoch} iter: {i} train_loss: {total_train_loss}")
                logging.info(f"epoch: {epoch} iter: {i} train_loss: {total_train_loss}")

        epoch_train_loss /= num_batches_train
        writer.add_scalar("Log/Train_loss", epoch_train_loss, epoch)
        train_loss_list.append(epoch_train_loss.detach().cpu().numpy())
        # # evaluation
        if epoch % args.val_interval == 0:
            
            inf_times = []
            
            with torch.no_grad():
                model.eval()
                all_comb = [f"A photo of a {class_name}." for class_name in test_dataloader.dataset.classes]
                text_evals = all_comb
                
                for i, batch in enumerate(tqdm(test_dataloader, total=num_batches_val)):
                    images, class_ids, image_paths = batch
                    
                    class_ids = class_ids.to(device)
                    images = images.to(device)
                    texts = [f"A photo of a {test_dataloader.dataset.classes[class_id]}." for class_id in class_ids]
                    texts = clip.tokenize(texts).to(device)

                    # 이미지별 예측 top 추출, 정답체크
                    text_eval_tokens = clip.tokenize(text_evals).to(device)
                    
                    st = time.time()
                    image_features = model.encode_image(images)
                    text_features = model.encode_text(text_eval_tokens)

                    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    logit_scale = model.logit_scale.exp()
                    logits_per_image_eval = logit_scale * image_features_norm @ text_features_norm.t()
                    logits_per_text_eval = logits_per_image_eval.t()
                    inf_times.append(time.time() - st)

                    text_probs = logits_per_image_eval.softmax(dim=-1)
                    top_probs, top_labels = text_probs.cpu().float().topk(2, dim=-1) # 예측 top5를 뽑음
                    
                    top5_text_evals = np.array(text_evals)[top_labels] # 사전에서 top5 인덱스 가져옴
                    top1_text_evals = top5_text_evals[:, 0]
                    
                    pred_list += [pred.split(' ')[-1][:-1] for pred in top1_text_evals]
                    gt_list += [test_dataloader.dataset.classes[class_id] for class_id in class_ids]
                    
                cf_matrix = confusion_matrix(gt_list, pred_list)
                accuracy = (np.sum(np.diag(cf_matrix)) / np.sum(cf_matrix)) * 100.
                print_confusion_matrix(cf_matrix, test_dataloader.dataset.classes)

                print(f"Epoch {epoch} accuracy : {accuracy}%")
                accuracy_list.append(accuracy)
                
                writer.add_scalar("Log/Test_acc", torch.tensor(accuracy), epoch)

                print(f"Epoch {epoch} inf time is {np.mean(inf_times)} seconds")
                print(f"Epoch {epoch} train loss: {epoch_train_loss}")
                train_loss_list.append(epoch_train_loss.cpu().numpy())
                logging.info(f"Epoch {epoch} train loss: {epoch_train_loss}")
                
        # save best accuracy
        if args.model_save_mode == 'acc':
            value_list = accuracy_list
            compare_value = accuracy
            if np.max(value_list) <= compare_value:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_train_loss,
                    }, weights_path / f"best_ep{epoch}_{args.model_save_mode}{np.round(accuracy, 2)}.pt")  #just change to your preferred folder/filename
                print(f"Saved weights under model_checkpoint/best_ep{epoch}_acc{np.round(accuracy, 2)}.pt.")
                logging.info(f"Saved weights under model_checkpoint/{weights_path}/best_ep{epoch}_{args.model_save_mode}{np.round(accuracy, 2)}.pt.")

        # save best loss
        elif args.model_save_mode == 'loss':
            value_list = train_loss_list
            compare_value = epoch_train_loss
            if np.min(value_list) >= compare_value:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_train_loss,
                    }, weights_path / f"best_ep{epoch}_{args.model_save_mode}{np.round(epoch_train_loss.detach().cpu().numpy(), 2)}.pt")  #just change to your preferred folder/filename
                print(f"Saved weights under model_checkpoint/best_ep{epoch}_loss{np.round(epoch_train_loss.detach().cpu().numpy(), 2)}.pt.")
                logging.info(f"Saved weights under model_checkpoint/{weights_path}/best_ep{epoch}_{args.model_save_mode}{np.round(epoch_train_loss.detach().cpu().numpy(), 2)}.pt.")

        else:
            raise ValueError
        
        

    writer.flush()
    writer.close()

