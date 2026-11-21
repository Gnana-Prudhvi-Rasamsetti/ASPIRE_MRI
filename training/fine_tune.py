"""Optional low-learning-rate end-to-end fine-tuning after frozen-expert fusion succeeds."""
from __future__ import annotations
import time
from pathlib import Path
import torch
from losses import StructurePreservingLoss
from .common import BatchProgress, log_early_stopping, log_epoch_summary, resume_checkpoint, save_checkpoint


def fine_tune(model,loader,validation_loader,device,epochs=10,learning_rate=1e-5,checkpoint_path='checkpoints/adaptive/fine_tuned.pt',latest_checkpoint_path=None,resume=False):
    for parameter in model.parameters(): parameter.requires_grad_(True)
    # train_adaptive freezes experts by design; this explicit loop is kept separate for controlled research use.
    return train_end_to_end(model,loader,validation_loader,device,epochs,learning_rate,checkpoint_path,latest_checkpoint_path,resume)


def train_end_to_end(model,loader,validation_loader,device,epochs,learning_rate,checkpoint_path,latest_checkpoint_path=None,resume=False):
    model.to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate); criterion=StructurePreservingLoss(); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=2,factor=.5); scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    default_latest_name='fine_tuned_latest.pt' if Path(checkpoint_path).stem=='fine_tuned' else 'latest.pt'
    latest_checkpoint_path=Path(latest_checkpoint_path or Path(checkpoint_path).with_name(default_latest_name))
    best,stale,start_epoch=float('inf'),0,1
    if resume:
        if not latest_checkpoint_path.exists(): raise FileNotFoundError(f'No resumable checkpoint found: {latest_checkpoint_path}')
        state=resume_checkpoint(latest_checkpoint_path,model,optimizer,scheduler,scaler,device)
        best,stale,start_epoch=state['best_validation_loss'],state['stale_epochs'],state['start_epoch']
        if stale>=5 or start_epoch>epochs: return best
    for epoch in range(start_epoch,epochs+1):
        epoch_start = time.perf_counter()
        model.train()
        train_values=[]
        with BatchProgress(loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for batch in pbar:
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'): out=model(batch['noisy'].to(device))
                loss=criterion(out['restored'].float(),batch['clean'].to(device))
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_item = loss.item()
                train_values.append(loss_item)
                pbar.set_postfix(loss=loss_item)

        model.eval()
        values=[]
        with torch.inference_mode():
            for batch in validation_loader:
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'): out=model(batch['noisy'].to(device))
                values.append(criterion(out['restored'].float(),batch['clean'].to(device)).item())
        validation_loss=sum(values)/len(values)
        train_loss = sum(train_values)/len(train_values) if train_values else 0.0
        scheduler.step(validation_loss)

        saved = False
        if validation_loss<best:
            best=validation_loss
            stale=0
            saved = True
            save_checkpoint(checkpoint_path,model,optimizer,epoch,{'training_loss':train_loss,'validation_loss':best},{'mode':'end_to_end','learning_rate':optimizer.param_groups[0]['lr']},scheduler,scaler,{'best_validation_loss':best,'stale_epochs':stale})
        else:
            stale+=1
        save_checkpoint(latest_checkpoint_path,model,optimizer,epoch,{'training_loss':train_loss,'validation_loss':validation_loss},{'mode':'end_to_end','learning_rate':optimizer.param_groups[0]['lr']},scheduler,scaler,{'best_validation_loss':best,'stale_epochs':stale})
        log_epoch_summary(
            epoch=epoch,
            epochs=epochs,
            duration_seconds=time.perf_counter() - epoch_start,
            train_loss=train_loss,
            val_loss=validation_loss,
            best_val_loss=best,
            saved=saved,
        )
        if stale>=5:
            log_early_stopping(epoch, 5)
            break
    return best
