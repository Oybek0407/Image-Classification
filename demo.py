# streamlit library
import torch, timm, pickle, argparse, streamlit as st
from PIL import Image
from torchvision import transforms as T

def run(args): 

    javob_nomlari_yulak = "fruits_answer_name.pickle"
    with open(f"{javob_nomlari_yulak}", "rb") as f: javob_nomlari = pickle.load(f) 
    javob_sonlari = len(javob_nomlari)
    
    mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
    tfs = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean = mean, std = std)])
    
    namuna = "sample/fresh_peach_1.jpg"

    model = model_yuklash(model_nomi = args.model_nomi, javob_sonlari = javob_sonlari, model_yulagi = args.model_yulagi)
    print(f"Train qilingan {args.model_nomi} model muvaffaqqiyatli yuklab olindi!")
    # print(m)
    st.title("Fruits Classifier")
    yulak = st.file_uploader("upload any fruits picture:")
    rasm, javob_nomi=bashorat(model = model, yulak = yulak, tfs =tfs, javob_nomlari = javob_nomlari) if yulak else bashorat(model = model, yulak = namuna, tfs =tfs, javob_nomlari = javob_nomlari)
        
    st.write("Yuklangan rasm :") ; st.image(rasm); st.write(f"Prediction name -> {javob_nomi.upper()}")
    # st.write("Yuklangan Rasm: "); st.image(rasm); st.write(f"Bashorat Javob -> {javob_nomi.upper()}")


def model_yuklash(model_nomi, javob_sonlari, model_yulagi):
    
    model= timm.create_model(model_name = model_nomi, num_classes = javob_sonlari)
    # model.torch.load(model_yulagi)
    model.load_state_dict(torch.load(model_yulagi))
    return model.eval()
        
def bashorat(model, yulak, tfs, javob_nomlari):
        rasm = Image.open(yulak).convert("RGB")
        # if isinstance(javob_nomlari, dict):
        #         javob_nomlari = list(javob_nomlari.keys())
        # else:
        #         javob_nomlari = javob_nomlari
        javob_nomlari = list(javob_nomlari.keys())  if isinstance(javob_nomlari, dict) else javob_nomlari
      
        bashorat_class = torch.argmax(model(tfs(rasm).unsqueeze(0)), dim =1).item()
        return rasm, javob_nomlari[bashorat_class]
                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Lentils Types Classification Demo")
    
    # Argument larni qo'shish
    parser.add_argument("-mn", "--model_nomi", type = str, default = "rexnet_150", help = "AI Model Nomi")
    parser.add_argument("-my", "--model_yulagi", type = str, default = "fruits_best/fruits_best_model.pht", help = "Trained model")
    
    args = parser.parse_args() 
    
    run(args)

    