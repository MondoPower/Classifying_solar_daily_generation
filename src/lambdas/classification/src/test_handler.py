from lambda_function import handler
#Aviva Communities

event = {"community_name": "Aviva Communities","report_type": "quarterly"}

result = handler(event, 'UnimportantTrash')

# print('-------RESULT-------')
# print(result)