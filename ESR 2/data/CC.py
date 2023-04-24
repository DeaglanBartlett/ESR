import csv
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from jax import random
import matplotlib.pyplot as plt
import corner
import arviz as az

with open('CC_Table1_2201.07241.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    data = [row for row in reader]
    data = data[1:]
    refs = [row[-1] for row in data]
    data = [row[:3] for row in data]

print("arXiv refs:")
refs = list(set(refs))
for r in refs:
    print(r)

z, Hobs, Herr = jnp.array(data, dtype=jnp.float32).transpose()

def model():
    Om0 = numpyro.sample('Om0', dist.Uniform(0,1))
    H0 = numpyro.sample('H0', dist.Uniform(50, 80))
    H = H0 * jnp.sqrt(Om0 * (1 + z) ** 3 + (1 - Om0))
    numpyro.sample("obs", dist.Normal(H, Herr), obs=Hobs)
    
# Sample
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
kernel = numpyro.infer.NUTS(model)
num_samples = 5000
mcmc = numpyro.infer.MCMC(kernel, num_warmup=1000, num_samples=num_samples)
mcmc.run(rng_key_)

mcmc.print_summary()
samples = mcmc.get_samples()

# Trace
res = az.from_numpyro(mcmc)
az.plot_trace(res, compact=True)
plt.tight_layout()

# Corner
fig2 = corner.corner(
    res,
    labels=list(samples.keys()),
    show_titles=True
)

# Prediction
posterior_H = (
                jnp.expand_dims(samples["H0"], -1) *
                jnp.sqrt(jnp.expand_dims(samples["Om0"], -1) * (1 + z) ** 3 +
                        (1 - jnp.expand_dims(samples["Om0"], -1)))
                )
mean_H = jnp.mean(posterior_H, axis=0)
hpdi_H = hpdi(posterior_H, 0.95)
idx = jnp.argsort(z)
x = z[idx]
y = Hobs[idx]
yerr = Herr[idx]
hpdi_y = hpdi_H[:,idx]
ymean = mean_H[idx]
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.plot(x, ymean, color='C1')
ax.fill_between(x, hpdi_y[0], hpdi_y[1], alpha=0.3, interpolate=True, color='C1')
ax.errorbar(x, y, yerr=yerr, fmt='.', markersize=5, zorder=-2, capsize=1, elinewidth=1, color='k')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$H \left( z \right) \ / \ {\rm km \, s^{-1} \, Mpc^{-1}}$')
ax.set_xlim(0, None)
ax.set_title('CC with 95% Confidence Band')
fig.tight_layout()

plt.show()


