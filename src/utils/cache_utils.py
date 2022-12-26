import os
import pickle
import utils.utils_os as os_utils


# Padronização no nome que a cache é salva 
# path: pasta origem onde está sendo pego as imagens  
# model_name: Qual modelo está sendo usado para treino 
# type_ouput: tipo de arquivo de saida
# suffix: uma palavras para indicar qual é o proposito deste arquivo
# mirrored : informar se foi ou não gerado imagens com espelhamento
# equalization : informar se foi ou não gerado imagens  esqualizada 
def __make_a_name_to_cache_training( path,model_name,type_ouput,suffix = '' ,mirrored=False, equalization=False ):
        # Função monta o nome e escreve o  modolo treinado em disco 
        filename = os.path.basename(path)
        filename = ''.join([filename,f'_{model_name}_'])

        filename = ''.join([filename,f'E{mirrored:d}_'])
        filename = ''.join([filename,f'H{equalization:d}_'])
        filename = ''.join([filename,f'{suffix}.{type_ouput}'])

        return filename


def __valid_folder():
    parent_dir = os_utils.get_src_projec()
    path = os.path.join(parent_dir,'src', '.cache')
    
    if not os.path.exists(path):
        # Cria a pasta caso ela não exista
        os.mkdir(path)
       

def save(path,model_name,data,type_ouput,suffix = '' ,mirrored=False, equalization=False):
    file_name = __make_a_name_to_cache_training(path,model_name,type_ouput,suffix ,mirrored, equalization)
    __valid_folder()
    
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, '.cache')
    file_name = os.path.join(path,file_name)

    pickle.dump(data, open(file_name, 'wb'))
    print(f'Modelo Salvo{file_name}')

def __make_a_name_to_cache_data_frame( path,type_ouput,suffix = ''  ):
        # Função monta o nome e escreve o  modolo treinado em disco 
        filename = os.path.basename(path)
        filename = ''.join([filename,f'{suffix}.{type_ouput}'])

        return filename


# Salva as imagens processadas e classificadas em um data frame 
def save_images_data_frame(path,data,suffix = '' ):
    file_name = __make_a_name_to_cache_data_frame(path,'data',suffix )
    __valid_folder()
    
    parent_dir = os_utils.get_src_projec()
    path = os.path.join(parent_dir,'src', '.cache')
    file_name = os.path.join(path,file_name)

    pickle.dump(data, open(file_name, 'wb'))
    print(f'Modelo Salvo{file_name}')
