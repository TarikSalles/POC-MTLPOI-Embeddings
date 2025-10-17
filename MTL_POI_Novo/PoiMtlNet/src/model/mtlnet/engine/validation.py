import torch
from sklearn.metrics import classification_report

from configs.globals import DEVICE

def validation_best_model(data_next,
                          data_category,
                          best_next,
                          best_category,
                          model):

    all_pred_next = []
    all_truth_next = []
    all_pred_category = []
    all_truth_category = []

    with torch.no_grad():
        model.load_state_dict(best_next)
        model.eval()
        for batch_next, batch_category in zip(data_next, data_category):
            x_next, y_next = batch_next['x'].to(DEVICE), batch_next['y'].to(DEVICE)
            x_category, _ = batch_category['x'].to(DEVICE), batch_category['y'].to(DEVICE)
            out_category, out_next = model((x_category, x_next))
            pred_next, truth_next = out_next, y_next
            pred_next_class = torch.argmax(pred_next, dim=1)
            all_pred_next.append(pred_next_class.cpu())
            all_truth_next.append(truth_next.cpu())

        model.load_state_dict(best_category)
        model.eval()
        for batch_next, batch_category in zip(data_next, data_category):
            x_next, _ = batch_next['x'].to(DEVICE), batch_next['y'].to(DEVICE)
            x_category, y_category = batch_category['x'].to(DEVICE), batch_category['y'].to(DEVICE)
            out_category, out_next = model((x_category, x_next))
            pred_category, truth_category = out_category, y_category
            pred_category_class = torch.argmax(pred_category, dim=1)
            all_pred_category.append(pred_category_class.cpu())
            all_truth_category.append(truth_category.cpu())


    pred_next = torch.cat(all_pred_next)
    truth_next = torch.cat(all_truth_next)
    pred_category = torch.cat(all_pred_category)
    truth_category = torch.cat(all_truth_category)

    # Generate classification reports
    report_next = classification_report(truth_next, pred_next, output_dict=True,zero_division=0)
    report_category = classification_report(truth_category, pred_category, output_dict=True,zero_division=0)
    return report_next, report_category




def validation_model(dataloader_category, dataloader_next, model, num_classes):
    """
    Perform final validation and add classification reports to training metrics.

    Args:
        dataloader_category: Dataloader for category prediction
        dataloader_next: Dataloader for next POI prediction
        model: The trained model
        num_classes: Number of POI classes
        training_metric: Training metrics object to update
    """
    with torch.no_grad():
        # Move validation data to device in batches to avoid OOM errors
        all_pred_next = []
        all_truth_next = []
        all_pred_category = []
        all_truth_category = []

    for batch_next, batch_category in zip(dataloader_next.val.dataloader, dataloader_category.val.dataloader):
        x_next, y_next = batch_next['x'].to(DEVICE), batch_next['y'].to(DEVICE)
        x_category, y_category = batch_category['x'].to(DEVICE), batch_category['y'].to(DEVICE)
        out_category, out_next = model((x_category, x_next))
        pred_next, truth_next = out_next, y_next
        pred_category, truth_category = out_category, y_category
        pred_next_class = torch.argmax(pred_next, dim=1)
        pred_category_class = torch.argmax(pred_category, dim=1)

        all_pred_next.append(pred_next_class.cpu())
        all_truth_next.append(truth_next.cpu())
        all_pred_category.append(pred_category_class.cpu())
        all_truth_category.append(truth_category.cpu())

    pred_next = torch.cat(all_pred_next)
    truth_next = torch.cat(all_truth_next)
    pred_category = torch.cat(all_pred_category)
    truth_category = torch.cat(all_truth_category)

    # Generate classification reports
    report_next = classification_report(truth_next, pred_next, output_dict=True)
    report_category = classification_report(truth_category, pred_category, output_dict=True)

    return report_next, report_category


def validation_model_by_head(dataloader_category, dataloader_next, model, num_classes):
    """
    Perform final validation and add classification reports to training metrics.

    Args:
        dataloader_category: Dataloader for category prediction
        dataloader_next: Dataloader for next POI prediction
        model: The trained model
        num_classes: Number of POI classes
        training_metric: Training metrics object to update
    """
    with torch.no_grad():
        # Move validation data to device in batches to avoid OOM errors
        all_pred_next = []
        all_truth_next = []
        all_pred_category = []
        all_truth_category = []

        # Evaluate next POI task in batches
        for batch in dataloader_next.val.dataloader:
            x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
            out = model.forward_next(x)
            pred, truth = out, y
            pred_class = torch.argmax(pred, dim=1)

            all_pred_next.append(pred_class.cpu())
            all_truth_next.append(truth.cpu())

        # Evaluate category task in batches
        for batch in dataloader_category.val.dataloader:
            x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
            out = model.forward_category(x)
            pred, truth = out, y
            pred_class = torch.argmax(pred, dim=1)

            all_pred_category.append(pred_class.cpu())
            all_truth_category.append(truth.cpu())

            # Concatenate results
        pred_next = torch.cat(all_pred_next)
        truth_next = torch.cat(all_truth_next)
        pred_category = torch.cat(all_pred_category)
        truth_category = torch.cat(all_truth_category)

        # Generate classification reports
        report_next = classification_report(truth_next, pred_next, output_dict=True)
        report_category = classification_report(truth_category, pred_category, output_dict=True)

        return report_next, report_category