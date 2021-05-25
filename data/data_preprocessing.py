import csv
import numpy as np
import math
import gensim
import math
from config import Config
from collections import Counter
from tqdm import tqdm, trange
import json
import sys
import re
np.set_printoptions(threshold=sys.maxsize)


#VERACITYlabels = {'None': [1.0, 0.0, 0.0, 0.0], 'unverified': [0.0, 1.0, 0.0, 0.0],'true': [0.0, 0.0, 1.0, 0.0], 'false': [0.0, 0.0, 0.0, 1.0]}

VERACITYlabels = {'true': [0.0, 1.0], 'false': [1.0, 0.0]}
config = Config()

class PhemeDataset():
    def __init__(self):
        self._wordsAndEmbeddings = {}
        self._senlen = config.seqLen
        self.all_words = None

    def read_alldata(self, filePath):
        with open(filePath,'r') as file_object:
            reader = json.load(file_object)
            text = []
            tweet_struct_temp = []
            text_style = []
            text_userVerified = []
            text_userGeo = []
            text_userScreenName = []
            text_userFollowersCounts = []
            text_userFriendsCounts = []
            text_userFavoritesCounts = []
            text_CommGeo = []
            text_CommSource = []
            text_CommFavorited =[]
            text_CommFavoriteCounts = []
            text_CommTextLen = []
            text_class = []
            text_id = []
            reactions = []
            veracityLabels = []
            comm_style = []
            userVerified = []
            userGeo = []
            userScreenName = []
            userFollowersCounts = []
            userFriendsCounts = []
            userFavoritesCounts = []
            CommGeo = []
            CommSource = []
            CommFavorited =[]
            CommFavoriteCounts = []
            CommTextLen = []
            CommId = []
            commId_all = []
            CommAdjMats = []
            Comm_chrono = []
            
            
            for Claim_id in reader:  
                if reader[Claim_id]['veracityLabel'] == 'None' or reader[Claim_id]['veracityLabel'] == 'unverified':
                    continue
                tweet_struct_temp.append(reader[Claim_id]['microFeas'])
                text.append(reader[Claim_id]['text'])
                text_style.append(reader[Claim_id]['style'])
                text_userVerified.append(reader[Claim_id]['text_userVerified'])
                text_userGeo.append(reader[Claim_id]['text_userGeo'])
                text_userScreenName.append(reader[Claim_id]['text_userScreenName'])
                text_userFollowersCounts.append(reader[Claim_id]['text_userFollowersCount'])
                text_userFriendsCounts.append(reader[Claim_id]['text_userFriendsCounts'])
                text_userFavoritesCounts.append(reader[Claim_id]['text_userFavoritesCounts'])
                text_CommGeo.append(reader[Claim_id]['text_Geo'])
                text_CommSource.append(reader[Claim_id]['text_Source'])
                text_CommFavorited.append(reader[Claim_id]['text_Favorited'])
                text_CommFavoriteCounts.append(reader[Claim_id]['text_FavoriteCount'])
                text_CommTextLen.append(reader[Claim_id]['text_TextLen'])
                text_class.append(reader[Claim_id]['class'])
                text_id.append(Claim_id)
                veracityLabels.append(reader[Claim_id]['veracityLabel'])

                if not reader[Claim_id].get('reaction'):
                    reactions.append([])
                    comm_style.append([])
                    userVerified.append([])
                    userGeo.append([])
                    userScreenName.append([])
                    userFollowersCounts.append([])
                    userFriendsCounts.append([])
                    userFavoritesCounts.append([])
                    CommGeo.append([])
                    CommSource.append([])
                    CommFavorited.append([])
                    CommFavoriteCounts.append([])
                    CommTextLen.append([])
                    CommId.append([])
                    commId_all.append([])
                    CommAdjMats.append((Claim_id,np.zeros((config.comm_size, config.comm_size))))
                    Comm_chrono.append([])
                else: 
                    reactions.append(reader[Claim_id]['reaction'])
                    comm_style.append(reader[Claim_id]['commStyle'])
                    userVerified.append(reader[Claim_id]['userVerified'])
                    userGeo.append(reader[Claim_id]['userGeo'])
                    userScreenName.append(reader[Claim_id]['userScreenName'])
                    userFollowersCounts.append(reader[Claim_id]['userFollowersCount'])
                    userFriendsCounts.append(reader[Claim_id]['userFriendsCounts'])
                    userFavoritesCounts.append(reader[Claim_id]['userFavoritesCounts'])
                    CommGeo.append(reader[Claim_id]['CommGeo'])
                    CommSource.append(reader[Claim_id]['CommSource'])
                    CommFavorited.append(reader[Claim_id]['CommFavorited'])
                    CommFavoriteCounts.append(reader[Claim_id]['CommFavoriteCount'])
                    CommTextLen.append(reader[Claim_id]['CommTextLen'])
                    adj_tmp = np.zeros((reader[Claim_id]['microFeas'][1], reader[Claim_id]['microFeas'][1]))
                    adj_mat_tmp, _ = self.build_adj(reader[Claim_id]['CommAdj']['structure'], adj_mat=adj_tmp, indexDict=reader[Claim_id]['CommAdj']['indexDict'])
                    CommId.append(reader[Claim_id]['commId'])
                    commId_all.append(reader[Claim_id]['CommAdj']['indexDict'])
                    CommAdjMats.append((Claim_id,adj_mat_tmp[1:,1:]))
                    Comm_chrono.append(reader[Claim_id]['mircor_temporal'])
                
            textAndReactions = []
            veracities = []

            for i in range(len(text)):
                if veracityLabels[i] not in veracities:
                    veracities.append(veracityLabels[i])

            vLabels = []
            for i in range(len(text)):
                textAndReact = text[i]  + " ".join(reactions[i])
                textAndReactions.append(textAndReact)
                vercLabel = VERACITYlabels[veracityLabels[i]]
                vLabels.append(vercLabel)

            claim_features = []
            for i in range(len(text)):
                claim_feature = [0]*config.nums_feat
                claim_feature[0] = int(text_userVerified[i])
                claim_feature[1] = int(text_userGeo[i])
                claim_feature[2] = int(text_userScreenName[i])
                claim_feature[3] = int(text_userFollowersCounts[i])
                claim_feature[4] = int(text_userFriendsCounts[i])
                claim_feature[5] = int(text_userFavoritesCounts[i])
                claim_feature[6] = int(text_CommGeo[i])
                claim_feature[7] = int(text_CommSource[i])
                claim_feature[8] = int(text_CommFavorited[i])
                claim_feature[9] = int(text_CommFavoriteCounts[i])
                claim_feature[10] = int(text_CommTextLen[i])
                claim_feature[11] = int(text_style[i][0])
                claim_feature[12] = int(text_style[i][1])
                claim_feature[13] = int(text_style[i][2])
                claim_feature[14] = int(text_style[i][3])
                claim_features.append(claim_feature)
            #print("text", len(text))
            #print("claim_features",len(claim_features))
            #print("claim_features", claim_features)
 
        textWords = [self._eStatementSplit(txt) for txt in text]

        #chronological order
        reactionWords = []
        for i in range(len(reactions)):
            reactionWords.append([])
            tmp_react = {}
            for j in range(len(reactions[i])):
                tmp_react[Comm_chrono[i][j]] = self._eStatementSplit(reactions[i][j])
            react_chrono = sorted(tmp_react.items(), reverse=False)
            for idx, comm in  react_chrono:
                reactionWords[-1].append(comm)                 
        comm_features = []
        comm_masks = [] 
        for i in range(len(reactions)):
            comm_features.append([])
            tmp_comm_fea = {}
            comm_masks.append(np.zeros((config.nums_feat)))
            for j in range(len(reactions[i])):
                comm_feature = [0]*config.nums_feat
                comm_feature[0] = int(userVerified[i][j])
                comm_feature[1] = int(userGeo[i][j])
                comm_feature[2] = int(userScreenName[i][j])
                comm_feature[3] = int(userFollowersCounts[i][j])
                comm_feature[4] = int(userFriendsCounts[i][j])
                comm_feature[5] = int(userFavoritesCounts[i][j])
                comm_feature[6] = int(CommGeo[i][j])
                comm_feature[7] = int(CommSource[i][j])
                comm_feature[8] = int(CommFavorited[i][j])
                comm_feature[9] = int(CommFavoriteCounts[i][j])
                comm_feature[10] = int(CommTextLen[i][j])
                comm_feature[11] = int(comm_style[i][j][0])
                comm_feature[12] = int(comm_style[i][j][1])
                comm_feature[13] = int(comm_style[i][j][2])
                comm_feature[14] = int(comm_style[i][j][3])
                tmp_comm_fea[Comm_chrono[i][j]] = comm_feature
                if j < config.nums_feat:
                        comm_masks[-1][j] += 1.
            comm_fea_chrono = sorted(tmp_comm_fea.items(), reverse=False)
            for idx, fea in  comm_fea_chrono:
                comm_features[-1].append(fea)  

        CommAdjMats_truncate = []
        
        for i in range(len(text)):
            if CommAdjMats[i][1].shape[0] > config.comm_size:
                CommAdjMat_truncate = CommAdjMats[i][1][:config.comm_size, :config.comm_size]
                CommAdjMats_truncate.append(CommAdjMat_truncate)
                continue
            CommAdjMat_truncate = np.zeros((config.comm_size, config.comm_size))
            for j in range(CommAdjMats[i][1].shape[0]):
                for k in range(CommAdjMats[i][1].shape[0]):
                    CommAdjMat_truncate[j][k] = CommAdjMats[i][1][j][k]
            CommAdjMats_truncate.append(CommAdjMat_truncate)
        CommAdjMats_truncate = CommAdjMats_truncate
        return textWords, reactionWords, vLabels, claim_features, comm_features, tweet_struct_temp, text_style, CommAdjMats_truncate, comm_masks, text_class, text_id

    def build_adj(self, aDict, adj_mat, indexDict={}, store=0, i=0):
        for  k in aDict:
            if isinstance(k, str):
                adj_mat[i][i]=1
                adj_mat[store][i]=1
                adj_mat[i][store]=1
                adj_mat, i = self.build_adj(aDict[k], adj_mat=adj_mat, indexDict=indexDict, store=indexDict[k], i=i+1)
            else:
                pass  
        return adj_mat, i

    def _eStatementSplit(self, statement):
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
        statement = re.sub(r'[{}]'.format('-'), ' ', statement)
        statement = statement.lower()
        words = statement.strip().split()
        words_update = ""
        for w in words:
            if not re.findall(r"http", w) and not re.findall(r"#", w) and not re.findall(r"@", w) and not re.findall(emoji_pattern,w) and not re.findall(r"#", w) and w != ":(" and w != ": (" and w != ":)" and w != ": )":   
                words_update += (' '+ w) 
        return words_update
    
    def get_struct_new(self, tweet_struct_temp):
        tweet_struct_new = []
        for tweet_struct in tweet_struct_temp:
                del tweet_struct[3]
                tweet_struct_new.append(tweet_struct)  
        return tweet_struct_new
    
    def get_comm_fea_truncs(self, comm_features, nums_comm):
        comm_features_trunc = []
        for i in range(len(comm_features)):
            comm_feature_trunc = []
            if len(comm_features[i]) < nums_comm:
                for comm_feature in comm_features[i]:
                    comm_feature_trunc.append(np.array(comm_feature))
                for j in range(nums_comm - len(comm_features[i])):
                    comm_feature_trunc.append(np.zeros((config.nums_feat)))
            else:
                for j in range(nums_comm):
                    comm_feature_trunc.append(comm_features[i][j])
            comm_features_trunc.append(np.array(comm_feature_trunc))
        return comm_features_trunc

