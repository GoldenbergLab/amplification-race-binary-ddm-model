facesddmcode = """
functions {
  real intensity2evidence(int e, int i, real w, real x, vector y, real z, vector ge1, vector ge2, vector ge3, vector ge4, vector gn1, vector gn2, vector gn3, vector gn4) {
    real deltac;
    real curve;
    deltac = 0;
    if (z == 2) {
      if (w == 1 && x == 1) {
        deltac += pow(y[e], ge1[i]);
      }
      else if (w == 2 && x == 1) {
        deltac += pow(y[e], ge2[i]);
      }
      else if (w == 1 && x == 2) {
        deltac += pow(y[e], ge3[i]);
      }
      else if (w == 2 && x == 2) {
        deltac += pow(y[e], ge4[i]);
      }
    }
    else {
      if (w == 1 && x == 1) {
        deltac += pow(y[e], -gn1[i]);
      }
      else if (w == 2 && x == 1) {
        deltac += pow(y[e], -gn2[i]);
      }
      else if (w == 1 && x == 2) {
        deltac += pow(y[e], -gn3[i]);
      }
      else if (w == 2 && x == 2) {
        deltac += pow(y[e], -gn4[i]);
      }
    }
    return deltac;
  }
}

data {
  int<lower=1> N;      // Number of subjects
  int<lower=1> T;      // Number of trials
  int<lower=1, upper=T> Tsub[N]; // number of trials per subject
  int<lower=-1, upper=2> choice[N, T]; // choice, (1) Non-Emotional, (2) Emotional
  real rt[N, T];       // rt
  real rtmin[N];       // minimum RT for each subject of the observed data
  real rtbound;        // lower bound or RT across all subjects (e.g., 0.1 second)
  vector[12] identity[N, T]; // vector of face identities for each trial, (1) Black, (2) White
  vector[12] intensity[N, T]; // vector of face intensities for each trial
  real valence[N, T]; // valence of the trial, (1) Happy, (2) Angry
}

parameters {
  // Hyper(group)-parameters
  vector[15] mu_pr; // prior means for (1) happy alpha, (2) angry alpha, (3) happy beta, (4) angry beta, (5) happy lambda, (6) angry lambda, (7) tau, (8) happy black gammaE, (9) happy white gammaE, (10) angry black gammaE, (11) angry white gammaE, (12) happy black gammaN, (13) happy white gammaN, (14) angry black gammaN, (15) angry white gammaN
  vector<lower=0>[15] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] alpha_pr[2];
  vector[N] zed_pr[2];
  vector[N] lambda_pr[2];
  vector[N] tau_pr;
  vector[N] beta_pr[4];
  vector[N] gamma_pr[4];
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0>[N]                alpha[2]; // boundary separation intercept, (1) Happy, (2) Angry
  vector<lower=0, upper=1>[N]       zed[2]; // initial bias, (1) Happy, (2) Angry
  vector<lower=0, upper=1>[N]       lambda[2]; // stochasticity of choice, (1) Happy, (2) Angry
  vector<lower=rtbound, upper=max(rtmin)>[N] tau; // nondecision time
  vector<lower=0, upper=1>[N]       beta[4]; // evidence curve for emotional, (1) Happy-Black, (2) Happy-White, (3) Angry-Black, (4) Angry-White
  vector<lower=0, upper=1>[N]       gamma[4]; // evidence curve for non-emotional, (1) Happy-Black, (2) Happy-White, (3) Angry-Black, (4) Angry-White

  for (i in 1:N) {
    zed[1][i] = Phi_approx(mu_pr[3] + sigma[3] * zed_pr[1][i]); // happy initial bias
    zed[2][i] = Phi_approx(mu_pr[4] + sigma[4] * zed_pr[2][i]); // angry initial bias
    lambda[1][i] = Phi_approx(mu_pr[5] + sigma[5] * lambda_pr[1][i]); // happy stochasticity
    lambda[2][i] = Phi_approx(mu_pr[6] + sigma[6] * lambda_pr[2][i]); // angry stochasticity
    tau[i]  = Phi_approx(mu_pr[7] + sigma[7] * tau_pr[i]) * (rtmin[i] - rtbound) + rtbound;
    beta[1] = Phi_approx(mu_pr[8] + sigma[8] * beta_pr[1]); // happy-black evidence emotional
    beta[2] = Phi_approx(mu_pr[9] + sigma[9] * beta_pr[2]); // happy-white evidence emotional
    beta[3] = Phi_approx(mu_pr[10] + sigma[10] * beta_pr[3]); // angry-black evidence emotional
    beta[4] = Phi_approx(mu_pr[11] + sigma[11] * beta_pr[4]); // angry-white evidence emotional
    gamma[1] = Phi_approx(mu_pr[12] + sigma[12] * gamma_pr[1]); // happy-black evidence non-emotional
    gamma[2] = Phi_approx(mu_pr[13] + sigma[13] * gamma_pr[2]); // happy-white evidence non-emotional
    gamma[3] = Phi_approx(mu_pr[14] + sigma[14] * gamma_pr[3]); // angry-black evidence non-emotional
    gamma[4] = Phi_approx(mu_pr[15] + sigma[15] * gamma_pr[4]); // angry-white evidence non-emotional
  }
  alpha[1] = exp(mu_pr[1] + sigma[1] * alpha_pr[1]); // happy boundary separation intercept
  alpha[2] = exp(mu_pr[2] + sigma[2] * alpha_pr[2]); // angry boundary separation intercept
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  // Individual parameters for non-centered parameterization
  alpha_pr[1] ~ normal(0, 1);
  alpha_pr[2] ~ normal(0, 1);
  zed_pr[1] ~ normal(0, 1);
  zed_pr[2] ~ normal(0, 1);
  lambda_pr[1] ~ normal(0, 1);
  lambda_pr[2] ~ normal(0, 1);
  tau_pr   ~ normal(0, 1);
  beta_pr[1]     ~ normal(0, 1);
  beta_pr[2]     ~ normal(0, 1);
  beta_pr[3]     ~ normal(0, 1);
  beta_pr[4]     ~ normal(0, 1);
  gamma_pr[1]     ~ normal(0, 1);
  gamma_pr[2]     ~ normal(0, 1);
  gamma_pr[3]     ~ normal(0, 1);
  gamma_pr[4]     ~ normal(0, 1);

  // Begin subject loop
  for (i in 1:N) {
    for (t in 1:(Tsub[i])) {
      real delta; // drift rate
      int eloop;
      delta = 0;
      eloop = 1;
      for (e in identity[i, t]) {
        delta += intensity2evidence(eloop, i, e, valence[i, t], intensity[i, t], choice[i, t], beta[1], beta[2], beta[3], beta[4], gamma[1], gamma[2], gamma[3], gamma[4]);
        eloop += 1;
      }
      // Response time distributed along wiener first passage time distribution
      if (choice[i, t] == 2 && valence[i, t] == 1) {
        rt[i, t] ~ wiener(alpha[1][i], tau[i], zed[1][i], lambda[1][i]*delta);
      }
      else if (choice[i, t] == 2 && valence[i, t] == 2) {
        rt[i, t] ~ wiener(alpha[2][i], tau[i], zed[2][i], lambda[2][i]*delta);
      }
      else if (choice[i, t] == 1 && valence[i, t] == 1) {
        rt[i, t] ~ wiener(alpha[1][i], tau[i], 1 - zed[1][i], -lambda[1][i]*delta);
      }
      else if (choice[i, t] == 1 && valence[i, t] == 2) {
        rt[i, t] ~ wiener(alpha[2][i], tau[i], 1 - zed[2][i], -lambda[2][i]*delta);
      }
    }
  } // end of subject loop
}

generated quantities {
  // For group level parameters
  real<lower=0>                         mu_alpha_happy; // boundary separation happy
  real<lower=0>                         mu_alpha_angry; // boundary separation angry
  real<lower=0, upper=1>                mu_zed_happy; // initial bias happy
  real<lower=0, upper=1>                mu_zed_angry; // initial bias angry
  real<lower=0, upper=1>                mu_lambda_happy; // stochasticity happy
  real<lower=0, upper=1>                mu_lambda_angry; // stochasticity angry
  real<lower=rtbound-0.01, upper=max(rtmin)> mu_tau; // nondecision time
  real<lower=0, upper=1>                mu_beta_black_happy; // evidence for emotional, black happy
  real<lower=0, upper=1>                mu_beta_white_happy; // evidence for emotional, white happy
  real<lower=0, upper=1>                mu_beta_black_angry; // evidence for emotional, black angry
  real<lower=0, upper=1>                mu_beta_white_angry; // evidence for emotional, white angry
  real<lower=0, upper=1>                mu_gamma_black_happy; // evidence for non-emotional, black happy
  real<lower=0, upper=1>                mu_gamma_white_happy; // evidence for non-emotional, white happy
  real<lower=0, upper=1>                mu_gamma_black_angry; // evidence for non-emotional, black angry
  real<lower=0, upper=1>                mu_gamma_white_angry; // evidence for non-emotional, white angry

  // For log likelihood calculation
  real log_lik[N];

  // Assign group level parameter values
  mu_alpha_happy = exp(mu_pr[1]); // boundary separation happy
  mu_alpha_angry = exp(mu_pr[2]); // boundary separation angry
  mu_zed_happy = Phi_approx(mu_pr[3]); // initial bias happy
  mu_zed_angry = Phi_approx(mu_pr[4]); // initial bias angry
  mu_lambda_happy = Phi_approx(mu_pr[5]); // stochasticity happy
  mu_lambda_angry = Phi_approx(mu_pr[6]); // stochasticity angry
  mu_tau = Phi_approx(mu_pr[7]) * (mean(rtmin)-rtbound) + rtbound; // nondecision time
  mu_beta_black_happy = Phi_approx(mu_pr[8]); // evidence for emotional, black happy
  mu_beta_white_happy = Phi_approx(mu_pr[9]); // evidence for emotional, white happy
  mu_beta_black_angry = Phi_approx(mu_pr[10]); // evidence for emotional, black angry
  mu_beta_white_angry = Phi_approx(mu_pr[11]); // evidence for emotional, white angry
  mu_gamma_black_happy = Phi_approx(mu_pr[12]); // evidence for non-emotional, black happy
  mu_gamma_white_happy = Phi_approx(mu_pr[13]); // evidence for non-emotional, white happy
  mu_gamma_black_angry = Phi_approx(mu_pr[14]); // evidence for non-emotional, black angry
  mu_gamma_white_angry = Phi_approx(mu_pr[15]); // evidence for non-emotional, white angry

  { // local section, this saves time and space
    // Begin subject loop
    for (i in 1:N) {
      log_lik[i] = 0;
      for (t in 1:(Tsub[i])) {
        real delta; // drift rate
        int eloop;
        delta = 0;
        eloop = 1;
        for (e in identity[i, t]) {
          delta += intensity2evidence(eloop, i, e, valence[i, t], intensity[i, t], choice[i, t], beta[1], beta[2], beta[3], beta[4], gamma[1], gamma[2], gamma[3], gamma[4]);
          eloop += 1;
        }
        if (choice[i, t] == 2 && valence[i, t] == 1) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[1][i], tau[i], zed[1][i], lambda[1][i]*delta);
        }
        else if (choice[i, t] == 2 && valence[i, t] == 2) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[2][i], tau[i], zed[2][i], lambda[2][i]*delta);
        }
        else if (choice[i, t] == 1 && valence[i, t] == 1) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[1][i], tau[i], 1 - zed[1][i], -lambda[1][i]*delta);
        }
        else if (choice[i, t] == 1 && valence[i, t] == 2) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[2][i], tau[i], 1 - zed[2][i], -lambda[2][i]*delta);
        }
      }
    }
  }
}
"""
