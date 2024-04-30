import os
import util.util as util

# 퍼센트 비율
TRAIN = 70
VALIDATION = 30
TEST = 0

DATA_SET_PATH = '/home/sky-server/Downloads/data_set/data/lv4_traffic'
SAVE_PATH     = '/home/sky-server/Downloads/data_set/save_output'
LOG_PATH      = '/home/sky-server/Downloads/data_set'

LABEL_EXT = '.json'
NAME_SPACE = None

PROCESS = 22
COUNT = -1

SAVE_VID_WIDTH = 1280
SAVE_VID_HEIGHT = 720
SAVE_VID_FPS = 15

def main():
    log_path = LOG_PATH
    if not log_path.endswith(os.sep):
        log_path = log_path + os.sep
    log_txt = open(log_path + 'convert_log.txt', 'w')

    p_train = TRAIN
    p_valid = VALIDATION
    p_test = TEST

    if not p_train + p_valid + p_test == 100:
        print('Data Segmentation Percentage Error : train - ', p_train, '%  validation - ', p_valid, '%  test - ', p_test, '%')
        log_txt.write('Data Segmentation Percentage Error : train - ' + str(p_train) + '%  validation - ' + str(p_valid) + '%  test - ' + str(p_test) + '%' + '\n')
        log_txt.close()
        return

    save_path = SAVE_PATH
    if not save_path.endswith(os.sep):
        save_path = save_path + os.sep

    train_path = save_path + 'train' + os.sep
    valid_path = save_path + 'valid' + os.sep
    test_path = save_path + 'test' + os.sep
    
    dataset_path = DATA_SET_PATH
    if not dataset_path.endswith(os.sep):
        dataset_path = dataset_path + os.sep
        
    print('dataset path : ' + dataset_path)
    log_txt.write('dataset path : ' + dataset_path + '\n')
    print('train dir : ' + train_path)
    log_txt.write('train dir : ' + train_path + '\n')
    print('val dir : ' + valid_path)
    log_txt.write('val dir : ' + valid_path + '\n')
    print('test dir : ' + test_path)
    log_txt.write('test dir : ' + test_path + '\n')

    label_extension = LABEL_EXT

    size = (SAVE_VID_WIDTH, SAVE_VID_HEIGHT)
    fps = SAVE_VID_FPS

    process = PROCESS
    ext_count = COUNT
    name_space = NAME_SPACE

    img_data, label_data = util.collect_data(dataset_path, label_extension)
    pair_list = util.data_classification(img_data, label_data, label_extension, process, ext_count)
    
    pair_len = len(pair_list)
    img_len = len(img_data)
    label_len = len(label_data)

    print('pair count :', pair_len, ' / img count :', img_len, '/ label count :', label_len)
    log_txt.write('pair count : ' + str(pair_len) +  ' / img count : ' + str(img_len) +  ' / label count : ' + str(label_len) + '\n')

    if not (pair_len > 0 and img_len > 0 and label_len > 0):
        print('Empty img or lable.')
        log_txt.write('Empty img or lable.\n')
        log_txt.close()
        return

    # if not (pair_len == min(img_len, label_len)):
    #     print('The number of labels or images differs from the number of paired data.')
    #     log_txt.write('The number of labels or images differs from the number of paired data.\n')
    #     log_txt.close()
    #     return

    ret, train_list, vaild_list, test_list = util.split_data(pair_list, p_train, p_valid, p_test)
    if not ret:
        log_txt.close()
        return

    print('train img count :', len(train_list), '/ vaild img count :', len(vaild_list), '/ test img count :', len(test_list))
    log_txt.write('train img count : ' + str(len(train_list)) + ' / vaild img count : ' + str(len(vaild_list)) + ' / test img count : ' + str(len(test_list)) + '\n')
    log_txt.write('\n')

    if not util.move_data(train_path, train_list, process, size, fps):
        print('fail move train data.')
        log_txt.write('fail move train data.\n')
        log_txt.close()
        return
    
    if not util.move_data(valid_path, vaild_list, process, size, fps):
        print('fail move valid data.')
        log_txt.write('fail move valid data.\n')
        log_txt.close()
        return

    if not util.move_data(test_path, test_list, process, size, fps):
        print('fail move test data.')
        log_txt.write('fail move test data.\n')
        log_txt.close()
        return
    
    print('Successful data partition and label to text conversion.')
    log_txt.write('Successful data partition and label to text conversion.\n')
    log_txt.close()
    
if __name__ == '__main__':
    main()