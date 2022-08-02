ddmracecode_freealpha_study1_2 = """
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
  real ratio[N, T];     // ratio of black to white
  int indexer[N, T];    // keeps track of specific groupings for parameter contrasts
}

parameters {
  // Hyper(group)-parameters
  vector[19] mu_pr; // prior means for (1) happy 25%B alpha, (2) happy 50%B alpha, (3) happy 75%B alpha, (4) angry 25% alpha, (5) angry 50% alpha, (6) angry 75% alpha, (7, 8, 9, 10, 11, 12) zed, (13, 14, 5, 16, 17, 18) delta, (19) tau
  vector<lower=0>[19] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] alpha_pr[6];
  vector[N] zed_pr[6];
  vector[N] delta_pr[6];
  vector[N] tau_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0>[N]                alpha[6]; // boundary separation intercept, (1) Happy25B, (2) Happy50B, (3) Happy75B, (4) Angry25B, (5) Angry50B, (6) Angry75B
  vector<lower=0, upper=1>[N]       zed[6]; // initial bias
  vector[N]                         delta[6]; // drift rate
  vector<lower=rtbound, upper=max(rtmin)>[N] tau; // nondecision time

  for (i in 1:N) {
    zed[1][i] = Phi_approx(mu_pr[7] + sigma[7] * zed_pr[1][i]);
    zed[2][i] = Phi_approx(mu_pr[8] + sigma[8] * zed_pr[2][i]);
    zed[3][i] = Phi_approx(mu_pr[9] + sigma[9] * zed_pr[3][i]);
    zed[4][i] = Phi_approx(mu_pr[10] + sigma[10] * zed_pr[4][i]);
    zed[5][i] = Phi_approx(mu_pr[11] + sigma[11] * zed_pr[5][i]);
    zed[6][i] = Phi_approx(mu_pr[12] + sigma[12] * zed_pr[6][i]);
    tau[i]  = Phi_approx(mu_pr[19] + sigma[19] * tau_pr[i]) * (rtmin[i] - rtbound) + rtbound;
  }
  alpha[1] = exp(mu_pr[1] + sigma[1] * alpha_pr[1]);
  alpha[2] = exp(mu_pr[2] + sigma[2] * alpha_pr[2]);
  alpha[3] = exp(mu_pr[3] + sigma[3] * alpha_pr[3]);
  alpha[4] = exp(mu_pr[4] + sigma[4] * alpha_pr[4]);
  alpha[5] = exp(mu_pr[5] + sigma[5] * alpha_pr[5]);
  alpha[6] = exp(mu_pr[6] + sigma[6] * alpha_pr[6]);
  delta[1] = mu_pr[13] + sigma[13] * delta_pr[1];
  delta[2] = mu_pr[14] + sigma[14] * delta_pr[2];
  delta[3] = mu_pr[15] + sigma[15] * delta_pr[3];
  delta[4] = mu_pr[16] + sigma[16] * delta_pr[4];
  delta[5] = mu_pr[17] + sigma[17] * delta_pr[5];
  delta[6] = mu_pr[18] + sigma[18] * delta_pr[6];
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  // Individual parameters for non-centered parameterization
  alpha_pr[1] ~ normal(0, 1);
  alpha_pr[2] ~ normal(0, 1);
  alpha_pr[3] ~ normal(0, 1);
  alpha_pr[4] ~ normal(0, 1);
  alpha_pr[5] ~ normal(0, 1);
  alpha_pr[6] ~ normal(0, 1);
  zed_pr[1] ~ normal(0, 1);
  zed_pr[2] ~ normal(0, 1);
  zed_pr[3] ~ normal(0, 1);
  zed_pr[4] ~ normal(0, 1);
  zed_pr[5] ~ normal(0, 1);
  zed_pr[6] ~ normal(0, 1);
  delta_pr[1] ~ normal(0, 1);
  delta_pr[2] ~ normal(0, 1);
  delta_pr[3] ~ normal(0, 1);
  delta_pr[4] ~ normal(0, 1);
  delta_pr[5] ~ normal(0, 1);
  delta_pr[6] ~ normal(0, 1);
  tau_pr   ~ normal(0, 1);

  // Begin subject loop
  for (i in 1:N) {
    for (t in 1:(Tsub[i])) {
      // Response time distributed along wiener first passage time distribution
      if (choice[i, t] == 2) {
        rt[i, t] ~ wiener(alpha[indexer[i, t]][i], tau[i], zed[indexer[i, t]][i], delta[indexer[i, t]][i]);
      }
      else if (choice[i, t] == 1) {
        rt[i, t] ~ wiener(alpha[indexer[i, t]][i], tau[i], 1 - zed[indexer[i, t]][i], -delta[indexer[i, t]][i]);
      }
    }
  } // end of subject loop
}

generated quantities {
  // For group level parameters
  real<lower=0>                         mu_alpha_happy_b25; // boundary separation happy b25
  real<lower=0>                         mu_alpha_happy_b50; // boundary separation happy b50
  real<lower=0>                         mu_alpha_happy_b75; // boundary separation happy b75
  real<lower=0>                         mu_alpha_angry_b25; // boundary separation angry b25
  real<lower=0>                         mu_alpha_angry_b50; // boundary separation angry b50
  real<lower=0>                         mu_alpha_angry_b75; // boundary separation angry b75
  real<lower=0, upper=1>                mu_zed_happy_b25; // initial bias happy b25
  real<lower=0, upper=1>                mu_zed_happy_b50; // initial bias happy b50
  real<lower=0, upper=1>                mu_zed_happy_b75; // initial bias happy b75
  real<lower=0, upper=1>                mu_zed_angry_b25; // initial bias angry b25
  real<lower=0, upper=1>                mu_zed_angry_b50; // initial bias angry b50
  real<lower=0, upper=1>                mu_zed_angry_b75; // initial bias angry b75
  real                                  mu_delta_happy_b25; // drift happy b25
  real                                  mu_delta_happy_b50; // drift happy b50
  real                                  mu_delta_happy_b75; // drift happy b75
  real                                  mu_delta_angry_b25; // drift angry b25
  real                                  mu_delta_angry_b50; // drift angry b50
  real                                  mu_delta_angry_b75; // drift angry b75
  real<lower=rtbound-0.01, upper=max(rtmin)> mu_tau; // nondecision time

  // For log likelihood calculation
  real log_lik[N];

  // Assign group level parameter values
  mu_alpha_happy_b25 = exp(mu_pr[1]); // boundary separation happy
  mu_alpha_happy_b50 = exp(mu_pr[2]);
  mu_alpha_happy_b75 = exp(mu_pr[3]);
  mu_alpha_angry_b25 = exp(mu_pr[4]);
  mu_alpha_angry_b50 = exp(mu_pr[5]);
  mu_alpha_angry_b75 = exp(mu_pr[6]);
  mu_zed_happy_b25 = Phi_approx(mu_pr[7]); // initial bias happy
  mu_zed_happy_b50 = Phi_approx(mu_pr[8]);
  mu_zed_happy_b75 = Phi_approx(mu_pr[9]);
  mu_zed_angry_b25 = Phi_approx(mu_pr[10]);
  mu_zed_angry_b50 = Phi_approx(mu_pr[11]);
  mu_zed_angry_b75 = Phi_approx(mu_pr[12]);
  mu_delta_happy_b25 = mu_pr[13]; // drift happy b25
  mu_delta_happy_b50 = mu_pr[14]; // drift b50
  mu_delta_happy_b75 = mu_pr[15]; // drift b75
  mu_delta_angry_b25 = mu_pr[16];
  mu_delta_angry_b50 = mu_pr[17];
  mu_delta_angry_b75 = mu_pr[18];
  mu_tau = Phi_approx(mu_pr[19]) * (mean(rtmin)-rtbound) + rtbound; // nondecision time

  { // local section, this saves time and space
    // Begin subject loop
    for (i in 1:N) {
      log_lik[i] = 0;
      for (t in 1:(Tsub[i])) {
        if (choice[i, t] == 2) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[indexer[i, t]][i], tau[i], zed[indexer[i, t]][i], delta[indexer[i, t]][i]);
        }
        else if (choice[i, t] == 1) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[indexer[i, t]][i], tau[i], 1 - zed[indexer[i, t]][i], -delta[indexer[i, t]][i]);
        }
      }
    }
  }
}
"""


ddmracecode_fixedalpha_study1_2 = """
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
  real ratio[N, T];     // ratio of black to white
  int indexer[N, T];    // keeps track of specific groupings for parameter contrasts
}

parameters {
  // Hyper(group)-parameters
  vector[14] mu_pr; // prior means for (1) alpha, (2, 3, 4, 5, 6, 7) zed, (8, 9, 10, 11, 12, 13) delta, (14) tau
  vector<lower=0>[14] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] alpha_pr;
  vector[N] zed_pr[6];
  vector[N] delta_pr[6];
  vector[N] tau_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0>[N]                alpha; // boundary separation intercept
  vector<lower=0, upper=1>[N]       zed[6]; // initial bias
  vector[N]                         delta[6]; // drift rate
  vector<lower=rtbound, upper=max(rtmin)>[N] tau; // nondecision time

  for (i in 1:N) {
    zed[1][i] = Phi_approx(mu_pr[2] + sigma[2] * zed_pr[1][i]);
    zed[2][i] = Phi_approx(mu_pr[3] + sigma[3] * zed_pr[2][i]);
    zed[3][i] = Phi_approx(mu_pr[4] + sigma[4] * zed_pr[3][i]);
    zed[4][i] = Phi_approx(mu_pr[5] + sigma[5] * zed_pr[4][i]);
    zed[5][i] = Phi_approx(mu_pr[6] + sigma[6] * zed_pr[5][i]);
    zed[6][i] = Phi_approx(mu_pr[7] + sigma[7] * zed_pr[6][i]);
    tau[i]  = Phi_approx(mu_pr[14] + sigma[14] * tau_pr[i]) * (rtmin[i] - rtbound) + rtbound;
  }
  alpha = exp(mu_pr[1] + sigma[1] * alpha_pr);
  delta[1] = mu_pr[8] + sigma[8] * delta_pr[1];
  delta[2] = mu_pr[9] + sigma[9] * delta_pr[2];
  delta[3] = mu_pr[10] + sigma[10] * delta_pr[3];
  delta[4] = mu_pr[11] + sigma[11] * delta_pr[4];
  delta[5] = mu_pr[12] + sigma[12] * delta_pr[5];
  delta[6] = mu_pr[13] + sigma[13] * delta_pr[6];
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  // Individual parameters for non-centered parameterization
  alpha_pr ~ normal(0, 1);
  zed_pr[1] ~ normal(0, 1);
  zed_pr[2] ~ normal(0, 1);
  zed_pr[3] ~ normal(0, 1);
  zed_pr[4] ~ normal(0, 1);
  zed_pr[5] ~ normal(0, 1);
  zed_pr[6] ~ normal(0, 1);
  delta_pr[1] ~ normal(0, 1);
  delta_pr[2] ~ normal(0, 1);
  delta_pr[3] ~ normal(0, 1);
  delta_pr[4] ~ normal(0, 1);
  delta_pr[5] ~ normal(0, 1);
  delta_pr[6] ~ normal(0, 1);
  tau_pr   ~ normal(0, 1);

  // Begin subject loop
  for (i in 1:N) {
    for (t in 1:(Tsub[i])) {
      // Response time distributed along wiener first passage time distribution
      if (choice[i, t] == 2) {
        rt[i, t] ~ wiener(alpha[i], tau[i], zed[indexer[i, t]][i], delta[indexer[i, t]][i]);
      }
      else if (choice[i, t] == 1) {
        rt[i, t] ~ wiener(alpha[i], tau[i], 1 - zed[indexer[i, t]][i], -delta[indexer[i, t]][i]);
      }
    }
  } // end of subject loop
}

generated quantities {
  // For group level parameters
  real<lower=0>                         mu_alpha; // boundary separation
  real<lower=0, upper=1>                mu_zed_happy_b25; // initial bias happy b25
  real<lower=0, upper=1>                mu_zed_happy_b50; // initial bias happy b50
  real<lower=0, upper=1>                mu_zed_happy_b75; // initial bias happy b75
  real<lower=0, upper=1>                mu_zed_angry_b25; // initial bias angry b25
  real<lower=0, upper=1>                mu_zed_angry_b50; // initial bias angry b50
  real<lower=0, upper=1>                mu_zed_angry_b75; // initial bias angry b75
  real                                  mu_delta_happy_b25; // drift happy b25
  real                                  mu_delta_happy_b50; // drift happy b50
  real                                  mu_delta_happy_b75; // drift happy b75
  real                                  mu_delta_angry_b25; // drift angry b25
  real                                  mu_delta_angry_b50; // drift angry b50
  real                                  mu_delta_angry_b75; // drift angry b75
  real<lower=rtbound-0.01, upper=max(rtmin)> mu_tau; // nondecision time

  // For log likelihood calculation
  real log_lik[N];

  // Assign group level parameter values
  mu_alpha = exp(mu_pr[1]); // boundary separation happy
  mu_zed_happy_b25 = Phi_approx(mu_pr[2]); // initial bias happy
  mu_zed_happy_b50 = Phi_approx(mu_pr[3]);
  mu_zed_happy_b75 = Phi_approx(mu_pr[4]);
  mu_zed_angry_b25 = Phi_approx(mu_pr[5]);
  mu_zed_angry_b50 = Phi_approx(mu_pr[6]);
  mu_zed_angry_b75 = Phi_approx(mu_pr[7]);
  mu_delta_happy_b25 = mu_pr[8]; // drift happy b25
  mu_delta_happy_b50 = mu_pr[9]; // drift b50
  mu_delta_happy_b75 = mu_pr[10]; // drift b75
  mu_delta_angry_b25 = mu_pr[11];
  mu_delta_angry_b50 = mu_pr[12];
  mu_delta_angry_b75 = mu_pr[13];
  mu_tau = Phi_approx(mu_pr[14]) * (mean(rtmin)-rtbound) + rtbound; // nondecision time

  { // local section, this saves time and space
    // Begin subject loop
    for (i in 1:N) {
      log_lik[i] = 0;
      for (t in 1:(Tsub[i])) {
        if (choice[i, t] == 2) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[i], tau[i], zed[indexer[i, t]][i], delta[indexer[i, t]][i]);
        }
        else if (choice[i, t] == 1) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[i], tau[i], 1 - zed[indexer[i, t]][i], -delta[indexer[i, t]][i]);
        }
      }
    }
  }
}
"""

ddmraceallcode_fixedalpha_study3 = """
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
  real ratio[N, T];     // ratio of black to white
  int indexer[N, T];    // keeps track of specific groupings for parameter contrasts
}

parameters {
  // Hyper(group)-parameters
  vector[22] mu_pr; // prior means for (1) alpha, (2, 3, 4, 5, 6, 7) zed, (8, 9, 10, 11, 12, 13) delta, (14) tau
  vector<lower=0>[22] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] alpha_pr;
  vector[N] zed_pr[10];
  vector[N] delta_pr[10];
  vector[N] tau_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0>[N]                alpha; // boundary separation intercept
  vector<lower=0, upper=1>[N]       zed[10]; // initial bias
  vector[N]                         delta[10]; // drift rate
  vector<lower=rtbound, upper=max(rtmin)>[N] tau; // nondecision time

  for (i in 1:N) {
    zed[1][i] = Phi_approx(mu_pr[2] + sigma[2] * zed_pr[1][i]);
    zed[2][i] = Phi_approx(mu_pr[3] + sigma[3] * zed_pr[2][i]);
    zed[3][i] = Phi_approx(mu_pr[4] + sigma[4] * zed_pr[3][i]);
    zed[4][i] = Phi_approx(mu_pr[5] + sigma[5] * zed_pr[4][i]);
    zed[5][i] = Phi_approx(mu_pr[6] + sigma[6] * zed_pr[5][i]);
    zed[6][i] = Phi_approx(mu_pr[7] + sigma[7] * zed_pr[6][i]);
    zed[7][i] = Phi_approx(mu_pr[8] + sigma[8] * zed_pr[7][i]);
    zed[8][i] = Phi_approx(mu_pr[9] + sigma[9] * zed_pr[8][i]);
    zed[9][i] = Phi_approx(mu_pr[10] + sigma[10] * zed_pr[9][i]);
    zed[10][i] = Phi_approx(mu_pr[11] + sigma[11] * zed_pr[10][i]);
    tau[i]  = Phi_approx(mu_pr[22] + sigma[22] * tau_pr[i]) * (rtmin[i] - rtbound) + rtbound;
  }
  alpha = exp(mu_pr[1] + sigma[1] * alpha_pr);
  delta[1] = mu_pr[12] + sigma[12] * delta_pr[1];
  delta[2] = mu_pr[13] + sigma[13] * delta_pr[2];
  delta[3] = mu_pr[14] + sigma[14] * delta_pr[3];
  delta[4] = mu_pr[15] + sigma[15] * delta_pr[4];
  delta[5] = mu_pr[16] + sigma[16] * delta_pr[5];
  delta[6] = mu_pr[17] + sigma[17] * delta_pr[6];
  delta[7] = mu_pr[18] + sigma[18] * delta_pr[7];
  delta[8] = mu_pr[19] + sigma[19] * delta_pr[8];
  delta[9] = mu_pr[20] + sigma[20] * delta_pr[9];
  delta[10] = mu_pr[21] + sigma[21] * delta_pr[10];
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  // Individual parameters for non-centered parameterization
  alpha_pr ~ normal(0, 1);
  zed_pr[1] ~ normal(0, 1);
  zed_pr[2] ~ normal(0, 1);
  zed_pr[3] ~ normal(0, 1);
  zed_pr[4] ~ normal(0, 1);
  zed_pr[5] ~ normal(0, 1);
  zed_pr[6] ~ normal(0, 1);
  zed_pr[7] ~ normal(0, 1);
  zed_pr[8] ~ normal(0, 1);
  zed_pr[9] ~ normal(0, 1);
  zed_pr[10] ~ normal(0, 1);
  delta_pr[1] ~ normal(0, 1);
  delta_pr[2] ~ normal(0, 1);
  delta_pr[3] ~ normal(0, 1);
  delta_pr[4] ~ normal(0, 1);
  delta_pr[5] ~ normal(0, 1);
  delta_pr[6] ~ normal(0, 1);
  delta_pr[7] ~ normal(0, 1);
  delta_pr[8] ~ normal(0, 1);
  delta_pr[9] ~ normal(0, 1);
  delta_pr[10] ~ normal(0, 1);
  tau_pr   ~ normal(0, 1);

  // Begin subject loop
  for (i in 1:N) {
    for (t in 1:(Tsub[i])) {
      // Response time distributed along wiener first passage time distribution
      if (choice[i, t] == 2) {
        rt[i, t] ~ wiener(alpha[i], tau[i], zed[indexer[i, t]][i], delta[indexer[i, t]][i]);
      }
      else if (choice[i, t] == 1) {
        rt[i, t] ~ wiener(alpha[i], tau[i], 1 - zed[indexer[i, t]][i], -delta[indexer[i, t]][i]);
      }
    }
  } // end of subject loop
}

generated quantities {
  // For group level parameters
  real<lower=0>                         mu_alpha; // boundary separation
  real<lower=0, upper=1>                mu_zed_happy_b0; // initial bias happy b0
  real<lower=0, upper=1>                mu_zed_happy_b25; // initial bias happy b25
  real<lower=0, upper=1>                mu_zed_happy_b50; // initial bias happy b50
  real<lower=0, upper=1>                mu_zed_happy_b75; // initial bias happy b75
  real<lower=0, upper=1>                mu_zed_happy_b1; // initial bias happy b1
  real<lower=0, upper=1>                mu_zed_angry_b0; // initial bias angry b0
  real<lower=0, upper=1>                mu_zed_angry_b25; // initial bias angry b25
  real<lower=0, upper=1>                mu_zed_angry_b50; // initial bias angry b50
  real<lower=0, upper=1>                mu_zed_angry_b75; // initial bias angry b75
  real<lower=0, upper=1>                mu_zed_angry_b1; // initial bias angry b1
  real                                  mu_delta_happy_b0; // drift happy b0
  real                                  mu_delta_happy_b25; // drift happy b25
  real                                  mu_delta_happy_b50; // drift happy b50
  real                                  mu_delta_happy_b75; // drift happy b75
  real                                  mu_delta_happy_b1; // drift happy b1
  real                                  mu_delta_angry_b0; // drift angry b0
  real                                  mu_delta_angry_b25; // drift angry b25
  real                                  mu_delta_angry_b50; // drift angry b50
  real                                  mu_delta_angry_b75; // drift angry b75
  real                                  mu_delta_angry_b1; // drift angry b1
  real<lower=rtbound-0.01, upper=max(rtmin)> mu_tau; // nondecision time

  // For log likelihood calculation
  real log_lik[N];

  // Assign group level parameter values
  mu_alpha = exp(mu_pr[1]); // boundary separation happy
  mu_zed_happy_b0 = Phi_approx(mu_pr[2]);
  mu_zed_happy_b25 = Phi_approx(mu_pr[3]); // initial bias happy
  mu_zed_happy_b50 = Phi_approx(mu_pr[4]);
  mu_zed_happy_b75 = Phi_approx(mu_pr[5]);
  mu_zed_happy_b1 = Phi_approx(mu_pr[6]);
  mu_zed_angry_b0 = Phi_approx(mu_pr[7]);
  mu_zed_angry_b25 = Phi_approx(mu_pr[8]);
  mu_zed_angry_b50 = Phi_approx(mu_pr[9]);
  mu_zed_angry_b75 = Phi_approx(mu_pr[10]);
  mu_zed_angry_b1 = Phi_approx(mu_pr[11]);
  mu_delta_happy_b0 = mu_pr[12];
  mu_delta_happy_b25 = mu_pr[13]; // drift happy b25
  mu_delta_happy_b50 = mu_pr[14]; // drift b50
  mu_delta_happy_b75 = mu_pr[15]; // drift b75
  mu_delta_happy_b1 = mu_pr[16];
  mu_delta_angry_b0 = mu_pr[17];
  mu_delta_angry_b25 = mu_pr[18];
  mu_delta_angry_b50 = mu_pr[19];
  mu_delta_angry_b75 = mu_pr[20];
  mu_delta_angry_b1 = mu_pr[21];
  mu_tau = Phi_approx(mu_pr[22]) * (mean(rtmin)-rtbound) + rtbound; // nondecision time

  { // local section, this saves time and space
    // Begin subject loop
    for (i in 1:N) {
      log_lik[i] = 0;
      for (t in 1:(Tsub[i])) {
        if (choice[i, t] == 2) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[i], tau[i], zed[indexer[i, t]][i], delta[indexer[i, t]][i]);
        }
        else if (choice[i, t] == 1) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[i], tau[i], 1 - zed[indexer[i, t]][i], -delta[indexer[i, t]][i]);
        }
      }
    }
  }
}
"""

ddmraceallcode_freealpha_study3 = """
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
  real ratio[N, T];     // ratio of black to white
  int indexer[N, T];    // keeps track of specific groupings for parameter contrasts
}

parameters {
  // Hyper(group)-parameters
  vector[31] mu_pr; // prior means for (1) alpha, (2, 3, 4, 5, 6, 7) zed, (8, 9, 10, 11, 12, 13) delta, (14) tau
  vector<lower=0>[31] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] alpha_pr[10];
  vector[N] zed_pr[10];
  vector[N] delta_pr[10];
  vector[N] tau_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0>[N]                alpha[10]; // boundary separation intercept
  vector<lower=0, upper=1>[N]       zed[10]; // initial bias
  vector[N]                         delta[10]; // drift rate
  vector<lower=rtbound, upper=max(rtmin)>[N] tau; // nondecision time

  for (i in 1:N) {
    for (j in 1:10) {
      zed[j][i] = Phi_approx(mu_pr[j+10] + sigma[j+10] * zed_pr[j][i]);
    }
    tau[i]  = Phi_approx(mu_pr[31] + sigma[31] * tau_pr[i]) * (rtmin[i] - rtbound) + rtbound;
  }
  for (j in 1:10) {
    alpha[j] = exp(mu_pr[j] + sigma[j] * alpha_pr[j]);
    delta[j] = mu_pr[j+20] + sigma[j+20] * delta_pr[j];
  }
}

model {
  // Hyperparameters
  mu_pr ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  // Individual parameters for non-centered parameterization
  for (j in 1:10) {
    alpha_pr[j] ~ normal(0, 1);
    zed_pr[j] ~ normal(0, 1);
    delta_pr[j] ~ normal(0, 1);
  }
  tau_pr   ~ normal(0, 1);

  // Begin subject loop
  for (i in 1:N) {
    for (t in 1:(Tsub[i])) {
      // Response time distributed along wiener first passage time distribution
      if (choice[i, t] == 2) {
        rt[i, t] ~ wiener(alpha[indexer[i, t]][i], tau[i], zed[indexer[i, t]][i], delta[indexer[i, t]][i]);
      }
      else if (choice[i, t] == 1) {
        rt[i, t] ~ wiener(alpha[indexer[i, t]][i], tau[i], 1 - zed[indexer[i, t]][i], -delta[indexer[i, t]][i]);
      }
    }
  } // end of subject loop
}

generated quantities {
  // For group level parameters
  real<lower=0>                         mu_alpha_happy_b0; // boundary separation happy b0
  real<lower=0>                         mu_alpha_happy_b25; // boundary separation happy b25
  real<lower=0>                         mu_alpha_happy_b50; // boundary separation happy b50
  real<lower=0>                         mu_alpha_happy_b75; // boundary separation happy b75
  real<lower=0>                         mu_alpha_happy_b100; // boundary separation happy b100
  real<lower=0>                         mu_alpha_angry_b0; // boundary separation angry b0
  real<lower=0>                         mu_alpha_angry_b25; // boundary separation angry b25
  real<lower=0>                         mu_alpha_angry_b50; // boundary separation angry b50
  real<lower=0>                         mu_alpha_angry_b75; // boundary separation angry b75
  real<lower=0>                         mu_alpha_angry_b100; // boundary separation angry b100
  real<lower=0, upper=1>                mu_zed_happy_b0; // initial bias happy b0
  real<lower=0, upper=1>                mu_zed_happy_b25; // initial bias happy b25
  real<lower=0, upper=1>                mu_zed_happy_b50; // initial bias happy b50
  real<lower=0, upper=1>                mu_zed_happy_b75; // initial bias happy b75
  real<lower=0, upper=1>                mu_zed_happy_b1; // initial bias happy b1
  real<lower=0, upper=1>                mu_zed_angry_b0; // initial bias angry b0
  real<lower=0, upper=1>                mu_zed_angry_b25; // initial bias angry b25
  real<lower=0, upper=1>                mu_zed_angry_b50; // initial bias angry b50
  real<lower=0, upper=1>                mu_zed_angry_b75; // initial bias angry b75
  real<lower=0, upper=1>                mu_zed_angry_b1; // initial bias angry b1
  real                                  mu_delta_happy_b0; // drift happy b0
  real                                  mu_delta_happy_b25; // drift happy b25
  real                                  mu_delta_happy_b50; // drift happy b50
  real                                  mu_delta_happy_b75; // drift happy b75
  real                                  mu_delta_happy_b1; // drift happy b1
  real                                  mu_delta_angry_b0; // drift angry b0
  real                                  mu_delta_angry_b25; // drift angry b25
  real                                  mu_delta_angry_b50; // drift angry b50
  real                                  mu_delta_angry_b75; // drift angry b75
  real                                  mu_delta_angry_b1; // drift angry b1
  real<lower=rtbound-0.01, upper=max(rtmin)> mu_tau; // nondecision time

  // For log likelihood calculation
  real log_lik[N];

  // Assign group level parameter values
  mu_alpha_happy_b0 = exp(mu_pr[1]); // boundary separation happy
  mu_alpha_happy_b25 = exp(mu_pr[2]);
  mu_alpha_happy_b50 = exp(mu_pr[3]);
  mu_alpha_happy_b75 = exp(mu_pr[4]);
  mu_alpha_happy_b100 = exp(mu_pr[5]);
  mu_alpha_angry_b0 = exp(mu_pr[6]);
  mu_alpha_angry_b25 = exp(mu_pr[7]);
  mu_alpha_angry_b50 = exp(mu_pr[8]);
  mu_alpha_angry_b75 = exp(mu_pr[9]);
  mu_alpha_angry_b100 = exp(mu_pr[10]);
  mu_zed_happy_b0 = Phi_approx(mu_pr[11]);
  mu_zed_happy_b25 = Phi_approx(mu_pr[12]); // initial bias happy
  mu_zed_happy_b50 = Phi_approx(mu_pr[13]);
  mu_zed_happy_b75 = Phi_approx(mu_pr[14]);
  mu_zed_happy_b1 = Phi_approx(mu_pr[15]);
  mu_zed_angry_b0 = Phi_approx(mu_pr[16]);
  mu_zed_angry_b25 = Phi_approx(mu_pr[17]);
  mu_zed_angry_b50 = Phi_approx(mu_pr[18]);
  mu_zed_angry_b75 = Phi_approx(mu_pr[19]);
  mu_zed_angry_b1 = Phi_approx(mu_pr[20]);
  mu_delta_happy_b0 = mu_pr[21];
  mu_delta_happy_b25 = mu_pr[22]; // drift happy b25
  mu_delta_happy_b50 = mu_pr[23]; // drift b50
  mu_delta_happy_b75 = mu_pr[24]; // drift b75
  mu_delta_happy_b1 = mu_pr[25];
  mu_delta_angry_b0 = mu_pr[26];
  mu_delta_angry_b25 = mu_pr[27];
  mu_delta_angry_b50 = mu_pr[28];
  mu_delta_angry_b75 = mu_pr[29];
  mu_delta_angry_b1 = mu_pr[30];
  mu_tau = Phi_approx(mu_pr[31]) * (mean(rtmin)-rtbound) + rtbound; // nondecision time

  { // local section, this saves time and space
    // Begin subject loop
    for (i in 1:N) {
      log_lik[i] = 0;
      for (t in 1:(Tsub[i])) {
        if (choice[i, t] == 2) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[indexer[i, t]][i], tau[i], zed[indexer[i, t]][i], delta[indexer[i, t]][i]);
        }
        else if (choice[i, t] == 1) {
          log_lik[i] += wiener_lpdf(rt[i, t] | alpha[indexer[i, t]][i], tau[i], 1 - zed[indexer[i, t]][i], -delta[indexer[i, t]][i]);
        }
      }
    }
  }
}
"""
