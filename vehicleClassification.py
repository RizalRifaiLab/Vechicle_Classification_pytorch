import torch
from torchvision import  transforms
from PIL import Image

import argparse

model1 = torch.load("cardataset_model_7.pt",map_location=torch.device('cpu'))
idx_to_class = {
    0: 'Ambulance', 1: 'Barge', 2: 'Bicycle', 3: 'Boat', 4: 'Bus', 5: 'Car', 6: 'Cart', 7: 'Caterpillar', 
    8: 'Helicopter', 9: 'Limousine', 10: 'Motorcycle', 11: 'Segway', 12: 'Snowmobile', 13: 'Tank', 14: 'Taxi', 15: 'Truck', 16: 'Van'}
def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''
    
    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_image = Image.open(test_image_name)

    
    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)
        for i in range(3):
            print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])


parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str)
args = parser.parse_args()

input_path = args.input_path
print(predict(model1, input_path))
#print(predict(model1, 'test/ambulance/ambulance.jpg'))