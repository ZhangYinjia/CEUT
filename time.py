import json

s1_fn = 'remote/20200318_kS1.log'
s2_fn = 'remote/20200318_kS2.log'

s1_log = {
    'cal_dis,ciphertext_calculation_time': 0.,
    'cal_dis,ciphertext_sign_time': 0.,
    'cal_dis,send/receive size': {
        'sent':0.,
        'received':0.
    },
    'cal_dis,socket_time': 0.,
    'find_min_dist,l=x,time': 0.,
    'reassign time': 0.,
    'find_min_dist,l=x,send/receive size': {
        'sent':0.,
        'received':0.
    },
    'find_min_dist,l=x,socket_time': 0.,
    'update_enc_center time': 0.
}

s2_log = {
    'cal_dist_s2_time': 0.,
    'find_min_dist,l=x,time': 0.,
}


with open(s1_fn) as s1fin:
    s1_lines = s1fin.readlines()

with open(s2_fn) as s2fin:
    s2_lines = s2fin.readlines()

s1_ln = 0
while s1_ln<len(s1_lines):
    line = s1_lines[s1_ln]
    kv=line.split(':')
    add3=False
    if 'cal_dis,ciphertext_calculation_time' in kv[0]:
        s1_log['cal_dis,ciphertext_calculation_time']+=eval(kv[1])
    elif 'cal_dis,ciphertext_sign_time' in kv[0]:
        s1_log['cal_dis,ciphertext_sign_time']+=eval(kv[1])
    elif 'cal_dis,send/receive size' in kv[0]:
        sent_line = s1_lines[s1_ln+1]
        s1_log['cal_dis,send/receive size']['sent']+=eval(sent_line.split(':')[1])
        received_line = s1_lines[s1_ln+2]
        s1_log['cal_dis,send/receive size']['received']+=eval(received_line.split(':')[1])
        add3=True
    elif 'cal_dis,socket_time' in kv[0]:
        s1_log['cal_dis,socket_time']+=eval(kv[1])
    elif 'find_min_dist,l=' in kv[0] and ',time' in kv[0]:
        s1_log['find_min_dist,l=x,time']+=eval(kv[1])
    elif 'reassign time' in kv[0]:
        s1_log['reassign time']+=eval(kv[1])
    elif 'find_min_dist,l=' in kv[0] and 'send/receive size' in kv[0]:
        sent_line = s1_lines[s1_ln+1]
        s1_log['find_min_dist,l=x,send/receive size']['sent']+=eval(sent_line.split(':')[1])
        received_line = s1_lines[s1_ln+2]
        s1_log['find_min_dist,l=x,send/receive size']['received']+=eval(received_line.split(':')[1])
        add3=True
    elif 'find_min_dist,l=' in kv[0] and 'socket_time' in kv[0]:
        s1_log['find_min_dist,l=x,socket_time']+=eval(kv[1])
        
    elif 'update_enc_center time' in kv[0]:
        s1_log['update_enc_center time']+=eval(kv[1])

    if add3:
        s1_ln+=3
    else:
        s1_ln+=1


s2_ln = 0
while s2_ln<len(s2_lines):
    line = s2_lines[s2_ln]
    kv = line.split(':')
    if 'find_min_dist' in kv[0]:
        s2_log['find_min_dist,l=x,time']+=eval(kv[1])
    elif 'cal_dist' in kv[0]:
        s2_log['cal_dist_s2_time']+=eval(kv[1])

    s2_ln+=1

print('(1,1): %f'%(s1_log['cal_dis,ciphertext_calculation_time']+s1_log['cal_dis,ciphertext_sign_time']))
print('(1,2): %f'%s2_log['cal_dist_s2_time'])
print('(1,3): %f'%(s1_log['cal_dis,send/receive size']['sent']+s1_log['cal_dis,send/receive size']['received']))
print('(1,4): %f'%(s1_log['cal_dis,socket_time']-s2_log['cal_dist_s2_time']))
print('(2,1): %f'%(s1_log['find_min_dist,l=x,time']+s1_log['reassign time']))
print('(2,2): %f'%(s2_log['find_min_dist,l=x,time']))
print('(2,3): %f'%(s1_log['find_min_dist,l=x,send/receive size']['sent']+s1_log['find_min_dist,l=x,send/receive size']['received']))
print('(2,4): %f'%(s1_log['find_min_dist,l=x,socket_time']-s2_log['find_min_dist,l=x,time']))
print('(3,1): %f'%(s1_log['update_enc_center time']))

