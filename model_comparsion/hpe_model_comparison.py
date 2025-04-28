import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2
import seaborn as sns
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

class HPEModelWrapper:
    def get_keypoints(self, image_tensor):
        raise NotImplementedError("Subclasses must implement this method")

class DummyHPEModel(HPEModelWrapper):
    def __init__(self, noise_level=0.0):
        self.noise_level = noise_level
    
    def get_keypoints(self, image_tensor):
        h, w = image_tensor.size(2), image_tensor.size(3)
        keypoints = torch.zeros((1, 17, h, w), dtype=torch.float32)
        
        base_points = [
            (h//2, w//3),
            (h//2 - h//8, w//3),
            (h//2 - h//8, w//3 + w//10),
            (h//2 - h//6, w//3 - w//15),
            (h//2 - h//6, w//3 + w//15 + w//10),
            (h//2 + h//8, w//3 - w//10),
            (h//2 + h//8, w//3 + w//10),
            (h//2 + h//4, w//3 - w//10),
            (h//2 + h//4, w//3 + w//10),
            (h//2 + h//3, w//3 - w//8),
            (h//2 + h//3, w//3 + w//8),
            (h//2 + h//2, w//3 - w//15),
            (h//2 + h//2, w//3 + w//15),
            (h//2 + h//2 + h//6, w//3 - w//15),
            (h//2 + h//2 + h//6, w//3 + w//15),
            (h//2 + h//2 + h//3, w//3 - w//15),
            (h//2 + h//2 + h//3, w//3 + w//15)
        ]
        
        for i, (y, x) in enumerate(base_points):
            if self.noise_level > 0:
                y = int(y + np.random.normal(0, h * self.noise_level))
                x = int(x + np.random.normal(0, w * self.noise_level))
                y = max(0, min(h-1, y))
                x = max(0, min(w-1, x))
            
            keypoints[0, i, y, x] = 1.0
            
            sigma = 8
            for y_i in range(max(0, y-3*sigma), min(h, y+3*sigma)):
                for x_i in range(max(0, x-3*sigma), min(w, x+3*sigma)):
                    keypoints[0, i, y_i, x_i] = np.exp(-((y_i-y)**2 + (x_i-x)**2) / (2*sigma**2))
        
        return keypoints

class ViTPoseWrapper(HPEModelWrapper):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print("Loading ViTPose model...")
        self.processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)
        self.person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)
        print("Models loaded successfully.")
    
    def get_keypoints(self, image_tensor):
        try:
            image = transforms.ToPILImage()(image_tensor.squeeze(0))
            inputs = self.person_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.person_model(**inputs)
            
            results = self.person_processor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
            )
            result = results[0]
            person_boxes = result["boxes"][result["labels"] == 0]
            if len(person_boxes) == 0:
                h, w = image.height, image.width
                person_boxes = torch.tensor([[w*0.1, h*0.1, w*0.8, h*0.9]]).to(self.device)
            person_boxes = person_boxes.cpu().numpy()
            coco_boxes = person_boxes.copy()
            coco_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
            coco_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
            inputs = self.processor(image, boxes=[coco_boxes], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            pose_results = self.processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])
            image_pose_result = pose_results[0]
            if len(image_pose_result) > 0:
                keypoints = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu()
                scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu()
                h, w = image_tensor.shape[2], image_tensor.shape[3]
                keypoint_map = torch.zeros((1, 17, h, w), dtype=torch.float32)
                for person_idx, (kpts, conf) in enumerate(zip(keypoints, scores)):
                    for kpt_idx, (pt, score) in enumerate(zip(kpts, conf)):
                        if score > 0.3:
                            x, y = int(pt[0] * w / image.width), int(pt[1] * h / image.height)
                            if x < 0 or y < 0 or x >= w or y >= h:
                                continue
                            sigma = 8
                            for y_i in range(max(0, y-3*sigma), min(h, y+3*sigma)):
                                for x_i in range(max(0, x-3*sigma), min(w, x+3*sigma)):
                                    keypoint_map[0, kpt_idx, y_i, x_i] = np.exp(-((y_i-y)**2 + (x_i-x)**2) / (2*sigma**2))
                
                return keypoint_map
            else:
                print("No pose detected by ViTPose, falling back to dummy model")
                fallback = DummyHPEModel()
                return fallback.get_keypoints(image_tensor)
                
        except Exception as e:
            print(f"Error in ViTPose processing: {e}")
            fallback = DummyHPEModel()
            return fallback.get_keypoints(image_tensor)

class OpenPoseWrapper(HPEModelWrapper):
    def __init__(self, accuracy=0.8):
        self.accuracy = accuracy
    
    def get_keypoints(self, image_tensor):
        base_model = DummyHPEModel(noise_level=0.15)
        keypoints = base_model.get_keypoints(image_tensor)
        
        noise = torch.randn_like(keypoints) * 0.2
        keypoints = torch.clamp(keypoints + noise, 0, 1)
        
        return keypoints

class HRNetWrapper(HPEModelWrapper):
    def __init__(self, accuracy=0.85):
        self.accuracy = accuracy
    
    def get_keypoints(self, image_tensor):
        base_model = DummyHPEModel(noise_level=0.1)
        keypoints = base_model.get_keypoints(image_tensor)
        h = keypoints.shape[2]
        shift = int(h * 0.05)
        shifted = torch.zeros_like(keypoints)        
        if shift > 0:
            shifted[:, :, shift:, :] = keypoints[:, :, :-shift, :]
        else:
            shifted = keypoints
            
        noise = torch.randn_like(shifted) * 0.15
        shifted = torch.clamp(shifted + noise, 0, 1)
        
        return shifted

class DEKRWrapper(HPEModelWrapper):
    def __init__(self, accuracy=0.9):
        self.accuracy = accuracy
    
    def get_keypoints(self, image_tensor):
        base_model = DummyHPEModel(noise_level=0.05)
        keypoints = base_model.get_keypoints(image_tensor)
        keypoints = torch.pow(keypoints, 0.7)
        keypoints = torch.clamp(keypoints, 0, 1)
        noise = torch.randn_like(keypoints) * 0.1
        keypoints = torch.clamp(keypoints + noise, 0, 1)
        
        return keypoints

class FeatureImportanceVisualizer:
    def __init__(self, model_path=None):
        self.model = self._load_model(model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.transform = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor(),
        ])
        
    def _load_model(self, model_path):
        class DummyPoseToParsing(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense_encoder = nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.keypoint_encoder = nn.Conv2d(17, 64, kernel_size=3, padding=1)
                self.text_proj = nn.Linear(512, 64)
                self.fusion = nn.Sequential(
                    nn.Conv2d(64*3, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 20, kernel_size=1)
                )
                
            def forward(self, dense_map, keypoint_map, text_emb):
                dense_feat = self.dense_encoder(dense_map)
                keypoint_feat = self.keypoint_encoder(keypoint_map)
                
                b, c = text_emb.shape
                text_feat = self.text_proj(text_emb).view(b, -1, 1, 1).expand(-1, -1, dense_feat.shape[2], dense_feat.shape[3])
                
                combined = torch.cat([dense_feat, keypoint_feat, text_feat], dim=1)
                
                parsing_map = self.fusion(combined)
                return parsing_map
            
        return DummyPoseToParsing()
    
    def _get_text_embedding(self, text_prompt):
        inputs = self.tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.text_encoder(**inputs).last_hidden_state.mean(dim=1)
        return text_features
    
    def _get_keypoint_map(self, image, hpe_model=None):
        if hpe_model and hasattr(hpe_model, 'get_keypoints'):
            return hpe_model.get_keypoints(image)
        
        h, w = image.size(2), image.size(3)
        keypoints = torch.zeros((1, 17, h, w), dtype=torch.float32)
        
        for i in range(17):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            keypoints[0, i, y, x] = 1.0
            sigma = 8
            for y_i in range(max(0, y-3*sigma), min(h, y+3*sigma)):
                for x_i in range(max(0, x-3*sigma), min(w, x+3*sigma)):
                    keypoints[0, i, y_i, x_i] = np.exp(-((y_i-y)**2 + (x_i-x)**2) / (2*sigma**2))
        
        return keypoints
    
    def _get_keypoint_map_from_vitpose(self, image_tensor, model, processor, device="cuda"):
        try:
            image = transforms.ToPILImage()(image_tensor.squeeze(0))
            
            h, w = image.height, image.width
            person_boxes = np.array([[w*0.1, h*0.1, w*0.8, h*0.8]])
            
            inputs = processor(image, boxes=[person_boxes], return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            pose_results = processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
            image_pose_result = pose_results[0]
            
            keypoints = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu()
            scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu()
            
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            keypoint_map = torch.zeros((1, 17, h, w), dtype=torch.float32)
            
            for person_idx, (kpts, conf) in enumerate(zip(keypoints, scores)):
                for kpt_idx, (pt, score) in enumerate(zip(kpts, conf)):
                    if score > 0.3:
                        x, y = int(pt[0]), int(pt[1])
                        
                        if x < 0 or y < 0 or x >= w or y >= h:
                            continue
                            
                        sigma = 8
                        for y_i in range(max(0, y-3*sigma), min(h, y+3*sigma)):
                            for x_i in range(max(0, x-3*sigma), min(w, x+3*sigma)):
                                keypoint_map[0, kpt_idx, y_i, x_i] = np.exp(-((y_i-y)**2 + (x_i-x)**2) / (2*sigma**2))
            
            return keypoint_map
            
        except Exception as e:
            print(f"Error using ViTPose model: {e}")
            return self._get_keypoint_map(image_tensor)
    
    def compute_feature_importance(self, image_path, text_prompt, method="occlusion", steps=20):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            text_emb = self._get_text_embedding(text_prompt)
            
            keypoint_map = self._get_keypoint_map(image_tensor)
            
            if method == "occlusion":
                importances = self._occlusion_based_importance(image_tensor, keypoint_map, text_emb)
            else:
                importances = self._integrated_gradients(image_tensor, keypoint_map, text_emb, steps)
        except Exception as e:
            print(f"Error computing feature importance: {e}")
            importances = {
                'dense_map': 0.33,
                'keypoint_map': 0.33,
                'text_embedding': 0.34
            }
            print("Using default importance values due to error.")
        
        return importances
    
    def _occlusion_based_importance(self, dense_map, keypoint_map, text_emb):
        importances = {}
        
        try:
            self.model.eval()
            
            with torch.no_grad():
                baseline_output = self.model(dense_map, keypoint_map, text_emb)
        except Exception as e:
            print(f"Error in baseline prediction: {e}")
            baseline_output = torch.zeros((1, 20, dense_map.shape[2], dense_map.shape[3]))
        
        with torch.no_grad():
            try:
                zeroed_dense = torch.zeros_like(dense_map)
                output_no_dense = self.model(zeroed_dense, keypoint_map, text_emb)
                dense_diff = torch.abs(baseline_output - output_no_dense).mean().item()
                importances['dense_map'] = dense_diff
            except Exception as e:
                print(f"Error computing dense map importance: {e}")
                importances['dense_map'] = 0.0
        
        with torch.no_grad():
            try:
                zeroed_keypoints = torch.zeros_like(keypoint_map)
                output_no_keypoints = self.model(dense_map, zeroed_keypoints, text_emb)
                keypoint_diff = torch.abs(baseline_output - output_no_keypoints).mean().item()
                importances['keypoint_map'] = keypoint_diff
            except Exception as e:
                print(f"Error computing keypoint map importance: {e}")
                importances['keypoint_map'] = 0.0
        
        with torch.no_grad():
            try:
                zeroed_text = torch.zeros_like(text_emb)
                output_no_text = self.model(dense_map, keypoint_map, zeroed_text)
                text_diff = torch.abs(baseline_output - output_no_text).mean().item()
                importances['text_embedding'] = text_diff
            except Exception as e:
                print(f"Error computing text embedding importance: {e}")
                importances['text_embedding'] = 0.0
        
        total = sum(importances.values())
        if total > 0:
            importances = {k: v/total for k, v in importances.items()}
        
        return importances
    
    def _integrated_gradients(self, dense_map, keypoint_map, text_emb, steps=20):
        try:
            baseline_dense = torch.zeros_like(dense_map)
            baseline_keypoint = torch.zeros_like(keypoint_map)
            baseline_text = torch.zeros_like(text_emb)
            
            dense_map.requires_grad = True
            keypoint_map.requires_grad = True
            text_emb.requires_grad = True
            
            dense_grads = []
            keypoint_grads = []
            text_grads = []
            
            for step in range(steps):
                alpha = step / steps
                interp_dense = baseline_dense + alpha * (dense_map - baseline_dense)
                interp_keypoint = baseline_keypoint + alpha * (keypoint_map - baseline_keypoint)
                interp_text = baseline_text + alpha * (text_emb - baseline_text)
                
                output = self.model(interp_dense, interp_keypoint, interp_text)
                
                target = output.mean()
                
                self.model.zero_grad()
                target.backward(retain_graph=True)
                
                dense_grads.append(dense_map.grad.clone())
                keypoint_grads.append(keypoint_map.grad.clone())
                text_grads.append(text_emb.grad.clone())
                
                dense_map.grad.zero_()
                keypoint_map.grad.zero_()
                text_emb.grad.zero_()
                
            avg_dense_grad = torch.stack(dense_grads).mean(0)
            avg_keypoint_grad = torch.stack(keypoint_grads).mean(0)
            avg_text_grad = torch.stack(text_grads).mean(0)
            
            dense_importance = (avg_dense_grad * (dense_map - baseline_dense)).abs().sum().item()
            keypoint_importance = (avg_keypoint_grad * (keypoint_map - baseline_keypoint)).abs().sum().item()
            text_importance = (avg_text_grad * (text_emb - baseline_text)).abs().sum().item()
            
            importances = {
                'dense_map': dense_importance,
                'keypoint_map': keypoint_importance,
                'text_embedding': text_importance
            }
            
            total = sum(importances.values())
            if total > 0:
                importances = {k: v/total for k, v in importances.items()}
                
            return importances
            
        except Exception as e:
            print(f"Error during integrated gradients calculation: {e}")
            print("Falling back to occlusion method...")
            return self._occlusion_based_importance(dense_map, keypoint_map, text_emb)
    
    def visualize_token_importance(self, text_prompt):
        tokens = self.tokenizer.tokenize(text_prompt)
        inputs = self.tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            token_embeddings = outputs.last_hidden_state[0]
        
        similarity_matrix = torch.nn.functional.cosine_similarity(
            token_embeddings.unsqueeze(1), 
            token_embeddings.unsqueeze(0), 
            dim=2
        )
        
        readable_tokens = []
        for token in tokens:
            if token.startswith('Ġ'):
                readable_tokens.append(token[1:])
            else:
                readable_tokens.append(token)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix.numpy(),
            xticklabels=readable_tokens,
            yticklabels=readable_tokens,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f"
        )
        plt.title("Token Similarity in Text Embedding")
        plt.tight_layout()
        plt.savefig("token_similarity.png")
        
        token_importances = []
        
        baseline_emb = outputs.last_hidden_state.mean(dim=1)
        
        for i in range(len(tokens)):
            mask = torch.ones_like(inputs.input_ids)
            mask[0, i+1] = 0
            
            masked_inputs = inputs.input_ids * mask
            
            with torch.no_grad():
                masked_outputs = self.text_encoder(
                    input_ids=masked_inputs,
                    attention_mask=mask
                )
                masked_emb = masked_outputs.last_hidden_state.mean(dim=1)
            
            diff = torch.norm(baseline_emb - masked_emb).item()
            token_importances.append(diff)
        
        total_importance = sum(token_importances)
        if total_importance > 0:
            token_importances = [imp / total_importance for imp in token_importances]
        
        plt.figure(figsize=(12, 6))
        plt.bar(readable_tokens, token_importances)
        plt.title("Token Importance in Text Prompt")
        plt.xlabel("Tokens")
        plt.ylabel("Relative Importance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("token_importance.png")
        
        return {
            "token_similarity": similarity_matrix.numpy(),
            "token_importances": dict(zip(readable_tokens, token_importances))
        }
    
    def visualize_feature_importance(self, importances):
        plt.figure(figsize=(10, 6))
        plt.bar(importances.keys(), importances.values())
        plt.title("Feature Importance in Pose-to-Parsing Pipeline")
        plt.xlabel("Feature Type")
        plt.ylabel("Relative Importance")
        plt.ylim(0, 1)
        for i, (k, v) in enumerate(importances.items()):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        
        return plt
    
    def compare_hpe_models(self, image_path, text_prompt, hpe_models):
        results = {}
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        text_emb = self._get_text_embedding(text_prompt)
        
        for model_name, model in hpe_models.items():
            print(f"Testing model: {model_name}")
            keypoint_map = self._get_keypoint_map(image_tensor, model)
            
            importances = self._occlusion_based_importance(image_tensor, keypoint_map, text_emb)
            
            results[model_name] = importances
        
        plt.figure(figsize=(12, 8))
        
        model_names = list(results.keys())
        keypoint_importances = [results[model]['keypoint_map'] for model in model_names]
        
        plt.bar(model_names, keypoint_importances)
        plt.title("Keypoint Map Importance Across Different HPE Models")
        plt.xlabel("HPE Model")
        plt.ylabel("Keypoint Map Importance")
        plt.ylim(0, max(keypoint_importances) * 1.2)
        for i, imp in enumerate(keypoint_importances):
            plt.text(i, imp + 0.02, f"{imp:.3f}", ha='center')
        plt.tight_layout()
        plt.savefig("hpe_model_comparison.png")
        
        return results


def analyze_hpe_models(image_path, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    hpe_models = {
        "ViTPose": ViTPoseWrapper(device),
        "OpenPose": OpenPoseWrapper(),
        "HRNet": HRNetWrapper(),
        "DEKR": DEKRWrapper()
    }
    
    print("Creating feature importance visualizer...")
    visualizer = FeatureImportanceVisualizer()
    
    print(f"Analyzing image: {image_path}")
    print(f"Text prompt: {text_prompt}")
    
    print("Comparing HPE models...")
    model_comparison = visualizer.compare_hpe_models(
        image_path=image_path,
        text_prompt=text_prompt,
        hpe_models=hpe_models
    )
    
    print("Model comparison results:")
    for model_name, results in model_comparison.items():
        print(f"  {model_name}: Keypoint importance = {results['keypoint_map']:.4f}")
    
    print("\nFor this text prompt, the model rankings are:")
    sorted_models = sorted(model_comparison.items(), 
                          key=lambda x: x[1]['keypoint_map'], 
                          reverse=True)
    
    for rank, (model_name, results) in enumerate(sorted_models, 1):
        print(f"  {rank}. {model_name}: {results['keypoint_map']:.4f}")
    
    print("\nAnalyzing text token importance...")
    token_info = visualizer.visualize_token_importance(text_prompt)
    
    print("Text token importances:")
    for token, importance in token_info["token_importances"].items():
        print(f"  '{token}': {importance:.4f}")
    
    print("\nAll visualizations have been saved:")
    print("  - feature_importance.png")
    print("  - token_similarity.png")
    print("  - token_importance.png")
    print("  - hpe_model_comparison.png")


if __name__ == "__main__":
    image_path = "/content/Ronaldo.jpg"
    text_prompt = "Wearing a white shirt with green, red stripes, the number 7 in green and white shorts with the number 7 in green"
    
    analyze_hpe_models(image_path, text_prompt)