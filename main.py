from finetune_model import ft_12lead_ECGFounder, retrain_12lead_model, load_12lead_ECGFounder_for_eval
from dataset import PediatricECGDataset, PediatricECGAgeDataset, process_labels
from util import find_optimal_threshold_optimized

def eval_model(
    model,
    dataloader,
    device,
    is_test=False,
    save_dir=None,
    n_classes=1,
    label_names=None,
    fixed_thresholds=None,
):

    model.eval()
    all_gts = []
    all_probs = []

    with torch.no_grad():
        for signals, labels in tqdm(dataloader):
            signals = signals.to(device)
            logits = model(signals)
            probs = torch.sigmoid(logits)

            all_gts.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_gts = np.concatenate(all_gts)
    all_probs = np.concatenate(all_probs)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        df_gts = pd.DataFrame(all_gts)
        df_gts.to_csv(os.path.join(save_dir, 'all_gts.csv'), index=False, float_format='%.5f')
        print('all_gts.csv saved')

        df_probs = pd.DataFrame(all_probs)
        df_probs.to_csv(os.path.join(save_dir, 'all_probs.csv'), index=False, float_format='%.5f')
        print('all_probs.csv saved')

    if all_gts.ndim == 1:
        all_gts = all_gts.reshape(-1, 1)
    if all_probs.ndim == 1:
        all_probs = all_probs.reshape(-1, 1)

    # =========================
    # 多标签分类评估
    # =========================
    if n_classes > 1:
        per_class_metrics = []

        if fixed_thresholds is None:
            best_thresholds = np.asarray(
                find_optimal_threshold_optimized(all_gts, all_probs),
                dtype=np.float64,
            ).reshape(-1)
            threshold_source = 'optimized_on_current_dataset'
        else:
            best_thresholds = np.asarray(fixed_thresholds, dtype=np.float64).reshape(-1)
            if len(best_thresholds) != n_classes:
                raise ValueError(
                    f"fixed_thresholds 长度为 {len(best_thresholds)}，"
                    f"但 n_classes={n_classes}，二者必须一致。"
                )
            best_thresholds = np.clip(best_thresholds, 0.0, 1.0)
            threshold_source = 'fixed_from_validation'

        all_preds = np.zeros_like(all_probs, dtype=np.int32)

        for class_idx in range(n_classes):
            best_thresh = float(best_thresholds[class_idx])
            class_preds = (all_probs[:, class_idx] >= best_thresh).astype(int)
            all_preds[:, class_idx] = class_preds

            tn, fp, fn, tp = confusion_matrix(
                all_gts[:, class_idx],
                class_preds,
                labels=[0, 1],
            ).ravel()

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            per_class_metrics.append({
                'class_index': class_idx,
                'label_name': label_names[class_idx] if label_names else f"Class_{class_idx}",
                'threshold': best_thresh,
                'precision': precision_score(
                    all_gts[:, class_idx], class_preds, zero_division=0
                ),
                'recall': recall_score(
                    all_gts[:, class_idx], class_preds, zero_division=0
                ),
                'specificity': specificity,
                'f1': f1_score(
                    all_gts[:, class_idx], class_preds, zero_division=0
                ),
                'auroc': roc_auc_score(
                    all_gts[:, class_idx], all_probs[:, class_idx]
                ) if len(np.unique(all_gts[:, class_idx])) > 1 else 0.0,
            })

        metrics = {
            'best_thresholds': best_thresholds,
            'threshold_source': threshold_source,
            'macro_auroc': np.mean([m['auroc'] for m in per_class_metrics]),
            'macro_f1': np.mean([m['f1'] for m in per_class_metrics]),
            'macro_precision': np.mean([m['precision'] for m in per_class_metrics]),
            'macro_recall': np.mean([m['recall'] for m in per_class_metrics]),
            'macro_specificity': np.mean([m['specificity'] for m in per_class_metrics]),
            'micro_f1': f1_score(
                all_gts.flatten(), all_preds.flatten(), zero_division=0
            ),
            'subset_accuracy': accuracy_score(all_gts, all_preds),
            'per_class_metrics': per_class_metrics,
            'auroc': np.mean([m['auroc'] for m in per_class_metrics]),
        }

        if is_test and save_dir:
            df_per_class = pd.DataFrame(per_class_metrics)
            csv_path = os.path.join(save_dir, 'per_class_performance.csv')
            df_per_class.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"逐类别结果已保存至: {csv_path}")

    # =========================
    # 二分类评估
    # =========================
    else:
        all_gts_1d = all_gts.flatten()
        all_probs_1d = all_probs.flatten()

        # 验证集：根据 ROC / Youden index 寻找阈值
        if fixed_thresholds is None:
            if len(np.unique(all_gts_1d)) < 2:
                best_thresh = 0.5
            else:
                fpr, tpr, thresholds = roc_curve(all_gts_1d, all_probs_1d)
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                best_thresh = float(np.clip(thresholds[best_idx], 0.0, 1.0))
            threshold_source = 'optimized_on_current_dataset'

        # 测试集：直接使用验证集确定的阈值
        else:
            threshold_array = np.asarray(fixed_thresholds, dtype=np.float64).reshape(-1)
            if threshold_array.size != 1:
                raise ValueError(
                    f"二分类 fixed_thresholds 应为单个数值，实际得到 {threshold_array.size} 个值。"
                )
            best_thresh = float(np.clip(threshold_array[0], 0.0, 1.0))
            threshold_source = 'fixed_from_validation'

        dynamic_preds = (all_probs_1d >= best_thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            all_gts_1d,
            dynamic_preds,
            labels=[0, 1],
        ).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        metrics = {
            'best_threshold': best_thresh,
            'threshold_source': threshold_source,
            'auroc': roc_auc_score(
                all_gts_1d, all_probs_1d
            ) if len(np.unique(all_gts_1d)) > 1 else 0.0,
            'acc': accuracy_score(all_gts_1d, dynamic_preds),
            'balanced_accuracy': (recall + specificity) / 2,
            'specificity': specificity,
            'f1': f1_score(all_gts_1d, dynamic_preds, zero_division=0),
            'recall': recall,
            'precision': precision_score(
                all_gts_1d, dynamic_preds, zero_division=0
            ),
        }

    return metrics


def save_validation_thresholds(thresholds, saved_dir, label_names=None):

    os.makedirs(saved_dir, exist_ok=True)

    threshold_array = np.asarray(thresholds, dtype=np.float64).reshape(-1)

    npy_path = os.path.join(saved_dir, 'validation_thresholds.npy')
    np.save(npy_path, threshold_array)

    if label_names is not None and len(label_names) == len(threshold_array):
        threshold_df = pd.DataFrame({
            'class_index': np.arange(len(threshold_array)),
            'label_name': label_names,
            'threshold': threshold_array,
        })
    else:
        threshold_df = pd.DataFrame({
            'class_index': np.arange(len(threshold_array)),
            'threshold': threshold_array,
        })

    csv_path = os.path.join(saved_dir, 'validation_thresholds.csv')
    threshold_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"验证集阈值已保存: {npy_path}")
    print(f"验证集阈值明细已保存: {csv_path}")

    return npy_path

def finetune():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_path = './wfdbdata'
    train_label_path = './csv/train.csv'
    test_label_path = "./csv/test_10000.csv"

    pretrained_pth = './checkpoint/12_lead_ECGFounder.pth'
    saved_dir = './res'
    os.makedirs(saved_dir, exist_ok=True)

    batch_size = 32
    lr = 1e-4
    epochs = 5

    n_classes, label_set, train_label_npy_path, test_label_npy_path = process_labels(
        train_label_path,test_label_path, saved_dir
    )

    train_dataset = PediatricECGDataset(
        data_path=data_path,
        label_path=train_label_npy_path,
        source_type='wfdb',
        csv_path=train_label_path,
        fs=500.0,
    )
    test_dataset = PediatricECGDataset(
        data_path=data_path,
        label_path=test_label_npy_path,
        source_type='wfdb',
        csv_path=test_label_path,
        fs=500.0,
    )


    val_size = int(0.2 * len(train_dataset))
    train_ds, val_ds = random_split(train_dataset, [len(train_dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ft_12lead_ECGFounder(device, pretrained_pth, n_classes, linear_prob=False)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            signals, labels = signals.to(device), labels.to(device)

            outputs = model(signals)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_m = eval_model(model, val_loader, device, n_classes=n_classes)

        if n_classes > 1:
            print(f"Val -> Macro AUC: {val_m['macro_auroc']:.4f}, "
                  f"Macro F1: {val_m['macro_f1']:.4f}, "
                  f"Subset Acc: {val_m['subset_accuracy']:.4f}")
            scheduler.step(val_m['macro_auroc'])
            current_auc = val_m['macro_auroc']
        else:
            print(f"Val -> AUC: {val_m['auroc']:.4f}, ACC: {val_m['acc']:.4f}, F1: {val_m['f1']:.4f}")
            scheduler.step(val_m['auroc'])
            current_auc = val_m['auroc']

        if current_auc > best_val_auc:
            best_val_auc = current_auc
            torch.save(model.state_dict(), os.path.join(saved_dir, 'best_finetune_model.pth'))
            print("==> Best Model Saved")

    best_model_path = os.path.join(saved_dir, 'best_finetune_model.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\n--- Determine Final Thresholds on Validation Set ---")
    best_val_m = eval_model(
        model,
        val_loader,
        device,
        n_classes=n_classes,
        label_names=label_set if n_classes > 1 else None,
        fixed_thresholds=None,
    )

    if n_classes > 1:
        final_thresholds = best_val_m['best_thresholds']
    else:
        final_thresholds = best_val_m['best_threshold']

    save_validation_thresholds(
        final_thresholds,
        saved_dir=saved_dir,
        label_names=label_set if n_classes > 1 else None,
    )

    print("\n--- Final Testing Performance (Fixed Validation Thresholds) ---")
    test_m = eval_model(
        model,
        test_loader,
        device,
        is_test=True,
        save_dir=saved_dir,
        n_classes=n_classes,
        label_names=label_set if n_classes > 1 else None,
        fixed_thresholds=final_thresholds,
    )

    if n_classes > 1:
        # 打印每个类别的指标
        print("\n--- Per-Class Metrics ---")

    else:
        print(f"Test AUROC:     {test_m['auroc']:.4f}")
        print(f"Test Accuracy:  {test_m['acc']:.4f}")
        print(f"Test Precision: {test_m['precision']:.4f}")
        print(f"Test Recall:    {test_m['recall']:.4f}")
        print(f"Test Specificity: {test_m['specificity']:.4f}")
        print(f"Test F1-Score:  {test_m['f1']:.4f}")

def test_only():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_path = './wfdbdata'
    test_label_path = "./csv/test_10000.csv"
    test_label_npy_path = './test_10000.npy'

    pretrained_pth = './checkpoint/12_lead_ECGFounder.pth'
    saved_dir = './res'
    os.makedirs(saved_dir, exist_ok=True)
    n_classes = None  # None：从 checkpoint 的 dense 输出维自动推断

    batch_size = 64
    threshold_path = None

    # === 加载全部数据（不划分）===
    full_dataset = PediatricECGDataset(
        data_path=data_path,
        label_path=test_label_npy_path,
        source_type='wfdb',
        csv_path=test_label_path,
        fs=500.0,
    )
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    # === 完整加载预训练权重（含分类头）===
    model, n_classes = load_12lead_ECGFounder_for_eval(
        device,
        pretrained_pth,
        n_classes=n_classes,
    )
    print(f"已加载完整权重，n_classes={n_classes}（与预训练分类头一致）")

    # === 准备固定阈值 ===
    if threshold_path is not None:
        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"阈值文件不存在: {threshold_path}")

        fixed_thresholds = np.load(threshold_path)
        fixed_thresholds = np.asarray(fixed_thresholds, dtype=np.float64).reshape(-1)

        expected_num = n_classes if n_classes > 1 else 1
        if len(fixed_thresholds) != expected_num:
            raise ValueError(
                f"阈值文件包含 {len(fixed_thresholds)} 个阈值，"
                f"但模型需要 {expected_num} 个。"
            )

        if n_classes == 1:
            fixed_thresholds = float(fixed_thresholds[0])

        print(f"使用内部验证集固定阈值: {threshold_path}")
    else:
        if n_classes > 1:
            fixed_thresholds = np.full(n_classes, 0.5, dtype=np.float64)
        else:
            fixed_thresholds = 0.5

        print(
            "警告：未提供内部验证集阈值文件。"
            "本次测试固定使用 0.5 阈值，不会在测试集上重新寻找最佳阈值。"
        )

    # === 对全部数据进行测试 ===
    test_metrics = eval_model(
        model,
        full_loader,
        device,
        is_test=True,
        save_dir=saved_dir,
        n_classes=n_classes,
        fixed_thresholds=fixed_thresholds,
    )

    # === 打印测试结果 ===
    if n_classes > 1:
        print("\n--- Per-Class Metrics ---")
    else:
        print(f"AUROC:     {test_metrics['auroc']:.4f}")
        print(f"Accuracy:  {test_metrics['acc']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall:    {test_metrics['recall']:.4f}")
        print(f"Test Specificity: {test_metrics['specificity']:.4f}")
        print(f"F1-Score:  {test_metrics['f1']:.4f}")


def retrain():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 数据准备
    data_path = './wfdbdata'
    train_label_path = './csv/train.csv'
    test_label_path = "./csv/test_10000.csv"

    saved_dir = './res'
    os.makedirs(saved_dir, exist_ok=True)

    batch_size = 64
    lr = 1e-4
    epochs = 20

    n_classes, label_set, train_label_npy_path, test_label_npy_path = process_labels(
        train_label_path,test_label_path, saved_dir
    )

    train_dataset = PediatricECGDataset(
        data_path=data_path,
        label_path=train_label_npy_path,
        source_type='wfdb',
        csv_path=train_label_path,
        fs=500.0,
    )
    test_dataset = PediatricECGDataset(
        data_path=data_path,
        label_path=test_label_npy_path,
        source_type='wfdb',
        csv_path=test_label_path,
        fs=500.0,
    )

    val_size = int(0.2 * len(train_dataset))
    train_ds, val_ds = random_split(train_dataset, [len(train_dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = retrain_12lead_model(n_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    best_val_auc = 0.0
    for epoch in range(epochs):
        model.train()
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_m = eval_model(model, val_loader, device, n_classes=n_classes)
        current_auc = val_m['macro_auroc'] if n_classes > 1 else val_m['auroc']
        scheduler.step(current_auc)

        if current_auc > best_val_auc:
            best_val_auc = current_auc
            torch.save(model.state_dict(), os.path.join(saved_dir, 'best_retrain_model.pth'))
            print(f"==> Saved Best Scratch Model (AUC: {current_auc:.4f})")

    best_model_path = os.path.join(saved_dir, 'best_retrain_model.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\n--- Determine Final Thresholds on Validation Set ---")
    best_val_m = eval_model(
        model,
        val_loader,
        device,
        n_classes=n_classes,
        label_names=label_set,
        fixed_thresholds=None,
    )

    if n_classes > 1:
        final_thresholds = best_val_m['best_thresholds']
    else:
        final_thresholds = best_val_m['best_threshold']

    save_validation_thresholds(
        final_thresholds,
        saved_dir=saved_dir,
        label_names=label_set if n_classes > 1 else None,
    )

    print("\n--- Final Testing Performance (Fixed Validation Thresholds) ---")
    test_m = eval_model(
        model,
        test_loader,
        device,
        is_test=True,
        save_dir=saved_dir,
        n_classes=n_classes,
        label_names=label_set,
        fixed_thresholds=final_thresholds,
    )

    if n_classes > 1:
        print(f"Scratch Training Final Macro AUC: {test_m['macro_auroc']:.4f}")
    else:
        print(f"Scratch Training Final AUC: {test_m['auroc']:.4f}")


if __name__ == "__main__":
    # test_only()
    finetune()
    # retrain()
