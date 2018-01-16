import praw

reddit = praw.Reddit(client_id='1f7S6FQFyX59yw',
                     client_secret='AwwfgxQlHWbYb9WzvOwlMMx6PVw',
                     password='windleaf5',
                     user_agent='testscript by /u/handsome-beaver',
                     username='handsome-beaver')


submissions = []
for submission in reddit.subreddit('askreddit').hot(limit=25):
    submissions.append(submission)


submission = submissions[0]
d = submissions[0].__dict__
for i, key in enumerate(d):
    attr = getattr(submission, key)
    print(key, attr)

file = open("testfile.txt",'w')

#for s in submissions:
#    file.write(s.title + ' ' + s.author)
#    file.write(s.body)

file.close()
