from __future__ import annotations
import time
from pathlib import Path
import torch
from losses import StructurePreservingLoss
from .common import BatchProgress, log_early_stopping, log_epoch_summary, resume_checkpoint, save_checkpoint


def train(model,loader,validation_loader,device,epochs=50,learning_rate=1e-4,checkpoint_path='checkpoints/adaptive/best.pt',patience=10,latest_checkpoint_path=None,resume=False):
    model.to(device); model.freeze_experts(); optimizer=torch.optim.AdamW(model.gate.parameters(),lr=learning_rate); criterion=StructurePreservingLoss(); scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,factor=.5)
    latest_checkpoint_path=Path(latest_checkpoint_path or Path(checkpoint_path).with_name('latest.pt'))
    best, stale, start_epoch=float('inf'),0,1
    if resume:
        if not latest_checkpoint_path.exists(): raise FileNotFoundError(f'No resumable checkpoint found: {latest_checkpoint_path}')
        state=resume_checkpoint(latest_checkpoint_path,model,optimizer,scheduler,scaler,device)
        best,stale,start_epoch=state['best_validation_loss'],state['stale_epochs'],state['start_epoch']
        model.freeze_experts()
        if stale>=patience or start_epoch>epochs: return best
    for epoch in range(start_epoch,epochs+1):
        epoch_start = time.perf_counter()
        model.train()
        model.freeze_experts()
        train_values=[]
        with BatchProgress(loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for batch in pbar:
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'): output=model(batch['noisy'].to(device))
                loss=criterion(output['restored'].float(),batch['clean'].to(device))
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_item = loss.item()
                train_values.append(loss_item)
                pbar.set_postfix(loss=loss_item)

        model.eval()
        model.freeze_experts()
        validation_values=[]
        with torch.inference_mode():
            for batch in validation_loader:
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'): output=model(batch['noisy'].to(device))
                validation_values.append(criterion(output['restored'].float(),batch['clean'].to(device)).item())
        validation_loss=sum(validation_values)/len(validation_values)
        train_loss = sum(train_values)/len(train_values) if train_values else 0.0
        scheduler.step(validation_loss)

        saved = False
        if validation_loss<best:
            best=validation_loss
            stale=0
            saved = True
            save_checkpoint(checkpoint_path,model,optimizer,epoch,{'training_loss':train_loss,'validation_loss':best},{'experts_frozen':True,'learning_rate':optimizer.param_groups[0]['lr']},scheduler,scaler,{'best_validation_loss':best,'stale_epochs':stale})
        else:
            stale+=1
        save_checkpoint(latest_checkpoint_path,model,optimizer,epoch,{'training_loss':train_loss,'validation_loss':validation_loss},{'experts_frozen':True,'learning_rate':optimizer.param_groups[0]['lr']},scheduler,scaler,{'best_validation_loss':best,'stale_epochs':stale})
        log_epoch_summary(
            epoch=epoch,
            epochs=epochs,
            duration_seconds=time.perf_counter() - epoch_start,
            train_loss=train_loss,
            val_loss=validation_loss,
            best_val_loss=best,
            saved=saved,
        )
        if stale>=patience:
            log_early_stopping(epoch, patience)
            break
    return best
