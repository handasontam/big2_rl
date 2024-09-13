import joblib
import pandas as pd
import matplotlib.pyplot as plt

losses = joblib.load('losses_th.pkl')
pg_loss = [l[0] for l in losses]
pd.Series(pg_loss).plot(style='.', ms=1)
pd.Series(pg_loss).rolling(1000).mean().plot()
plt.savefig('./plot_pg.png')
plt.close()

v_loss = [l[1] for l in losses]
pd.Series(v_loss).plot(style='.', ms=1)
pd.Series(v_loss).rolling(1000).mean().plot()
#plt.yscale('log')
plt.savefig('./plot_v.png')
plt.close()

ent_loss = [l[2] for l in losses]
pd.Series(ent_loss).plot(style='.', ms=1)
pd.Series(ent_loss).rolling(1000).mean().plot()
plt.savefig('./plot_ent.png')


