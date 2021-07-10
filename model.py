import sys
import numpy as np
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import matplotlib.pyplot as plt


class UKFSIRD:
    skip_empty_columns = 4
    variable_count = 7 # Number of state variables to filter (I, R, D, β, γ, μ and n, the population).
    measured_count = 3 # Number of measured variables (I, R and D).
    def __init__(self, country = "Vietnam", n = -1):
        self.confirmed_url = 'https://bit.ly/35yJO0d'
        self.recovered_url = 'https://bit.ly/2L6jLE9'
        self.deaths_url = 'https://bit.ly/2L0hzxQ'
        self.population_url = 'https://bit.ly/2WYjZCD'

        self.country = country

        self.first_date = "2020-01-22"
        self.last_date = None
        self.predicted_last_date = None

        self.data = None
        self.population = None

        self.number_days = n
        self.x = None

        self.ukf = None

        self.beta = None
        self.gamma = None
        self.mu = None

        self.curday = 0 

        self.predicted_beta_values = None
        self.predicted_gamma_values = None
        self.predicted_mu_values = None

        self.data_s_values = None
        self.data_i_values = None
        self.data_r_values = None
        self.data_d_values = None

        self.predicted_s_values = None
        self.predicted_i_values = None
        self.predicted_r_values = None
        self.predicted_d_values = None

    def pull_data(self):
        confirmed_data, confirmed_data_start = self.get_data(self.confirmed_url, self.country)
        recovered_data, recovered_data_start = self.get_data(self.recovered_url, self.country)
        deaths_data, deaths_data_start = self.get_data(self.deaths_url, self.country)
        data_start = min(confirmed_data_start, recovered_data_start, deaths_data_start)

        for i in range(data_start, confirmed_data.shape[1]):
            c = confirmed_data.iloc[0][i]
            r = recovered_data.iloc[0][i]
            d = deaths_data.iloc[0][i]
            data = [c - r - d, r, d]

            if self.data is None:
                self.data = np.array(data)
            else:
                self.data = np.vstack((self.data, data))

        if self.number_days > 0:
            self.data = self.data[:number_days]

        response = requests.get(self.population_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        data = soup.select('div div div div div tbody tr')

        populations = {}
        for i in range(len(data)):
            country_soup = BeautifulSoup(data[i].prettify(), 'html.parser')
            country_value = country_soup.select('tr td a')[0].get_text().strip()
            population_value = country_soup.select('tr td')[2].get_text().strip().replace(',', '')
            populations[country_value] = int(population_value)
        if self.country in populations:
            self.population = populations[self.country]
        else:
            sys.exit('Error: no population data is available for {}.'.format(self.country))

        self.reset()
    
    @staticmethod
    def get_data(url, country):
        data = pd.read_csv(url)
        data = data[(data['Country/Region'] == country) & data['Province/State'].isnull()]
        if data.shape[0] == 0:
            sys.exit('Error: no Covid-19 data is available for {}.'.format(country))

        data = data.drop(data.columns[list(range(SIRD.skip_empty_columns))], axis=1)  # Skip non-data columns.
        start = None

        for i in range(data.shape[1]):
            if data.iloc[0][i] != 0:
                start = SIRD.skip_empty_columns + i

                break

        return data, start
    
    def at_day(self, index):
        return self.data[index] if self.data is not None else math.nan

    def s_at_day(self, day):
        if self.data is not None:
            return self.population - self.i_at_day(day) - self.r_at_day(day) - self.d_at_day(day)
        return  math.nan

    def i_at_day(self, day):
        return self.at_day(day)[0]

    def r_at_day(self, day):
        return self.at_day(day)[1]

    def d_at_day(self, day):
        return self.at_day(day)[2]

    def predicted_s_value(self):
        return self.n - self.x.sum()

    def predicted_i_value(self):
        return self.x[0]

    def predicted_r_value(self):
        return self.x[1]

    def predicted_d_value(self):
        return self.x[2]

    #State variables calculate function
    @staticmethod                                
    def f(x, dt, **kwargs):

        model_self = kwargs.get('model_self')

        s = x[6] - x[:3].sum()
        beta = x[3]
        gamma = x[4]
        mu = x[5]
        n = x[6]

        a = np.array([
                     [1 + dt*(beta*s/n-gamma-mu),0, 0, 0, 0, 0, 0],
                     [dt*gamma,                  1, 0, 0, 0, 0, 0],
                     [dt*mu,                     0, 1, 0, 0, 0, 0],
                     [0,                         0, 0, 1, 0, 0, 0],
                     [0,                         0, 0, 0, 1, 0, 0],
                     [0,                         0, 0, 0, 0, 1, 0],
                     [0,                         0, 0, 0, 0, 0, 1]])
        return a @ x

    #Mesured variables
    @staticmethod
    def h(x):
        return x[:SIRD.measured_count]

    def reset(self):

        self.beta = 0.4
        self.gamma = 0.035
        self.mu = 0.005

        self.x = np.array([self.i_at_day(0), self.r_at_day(0), self.d_at_day(0)])
        self.n = self.population

        points = MerweScaledSigmaPoints(SIRD.variable_count, 1e-3, 2, 0)
        self.ukf = UnscentedKalmanFilter(SIRD.variable_count, SIRD.measured_count, 1, self.h, self.f, points)
        self.ukf.x = np.array([self.i_at_day(0), self.r_at_day(0), self.d_at_day(0), self.beta, self.gamma, self.mu, self.n])

        self.data_s_values = np.array([self.s_at_day(0)])
        self.data_i_values = np.array([self.i_at_day(0)])
        self.data_r_values = np.array([self.r_at_day(0)])
        self.data_d_values = np.array([self.d_at_day(0)])

        self.predicted_s_values = np.array([self.predicted_s_value()])
        self.predicted_i_values = np.array([self.predicted_i_value()])
        self.predicted_r_values = np.array([self.predicted_r_value()])
        self.predicted_d_values = np.array([self.predicted_d_value()])

        self.predicted_beta_values = np.array([self.beta])
        self.predicted_gamma_values = np.array([self.gamma])
        self.predicted_mu_values = np.array([self.mu])

    def solve(self, nb_of_days=None):
        max_days = self.data[:,1].shape[0]
        if nb_of_days is None or nb_of_days > max_days or nb_of_days <= 0:
            nb_of_days = max_days

        self.curday = nb_of_days

        for i in range(nb_of_days):
            self.ukf.predict(model_self=self)
            self.ukf.update(np.array([self.i_at_day(i), self.r_at_day(i), self.d_at_day(i)]))

            self.x = self.ukf.x[:3]
            self.beta = self.ukf.x[3]
            self.gamma = self.ukf.x[4]
            self.mu = self.ukf.x[5]
        
            self.data_s_values = np.append(self.data_s_values, self.s_at_day(i))
            self.data_i_values = np.append(self.data_i_values, self.i_at_day(i))
            self.data_r_values = np.append(self.data_r_values, self.r_at_day(i))
            self.data_d_values = np.append(self.data_d_values, self.d_at_day(i))

            self.predicted_s_values = np.append(self.predicted_s_values, self.predicted_s_value())
            self.predicted_i_values = np.append(self.predicted_i_values, self.predicted_i_value())
            self.predicted_r_values = np.append(self.predicted_r_values, self.predicted_r_value())
            self.predicted_d_values = np.append(self.predicted_d_values, self.predicted_d_value())

            self.predicted_beta_values = np.append(self.predicted_beta_values, self.beta)
            self.predicted_gamma_values = np.append(self.predicted_gamma_values, self.gamma)
            self.predicted_mu_values = np.append(self.predicted_mu_values, self.mu)

    def original_pred(self, nb_of_days = None):
        max_days = self.data[:,1].shape[0]
        if nb_of_days is None or nb_of_days > max_days or nb_of_days <= 0:
            nb_of_days = max_days

        for i in range(nb_of_days):
            prev_s = self.predicted_s_value()
            prev_i = self.predicted_i_value()

            s = (1 - self.beta*prev_i/self.n)*prev_s
            i = (1 - self.gamma + self.beta*prev_s/self.n)*prev_i - self.mu*prev_i
            r = self.gamma*prev_i + self.predicted_r_value()
            d = self.mu*prev_i + self.predicted_d_value()

            self.x = np.array([i, r, d])
            #print(self.x)

            self.predicted_s_values = np.append(self.predicted_s_values, self.predicted_s_value())
            self.predicted_i_values = np.append(self.predicted_i_values, self.predicted_i_value())
            self.predicted_r_values = np.append(self.predicted_r_values, self.predicted_r_value())
            self.predicted_d_values = np.append(self.predicted_d_values, self.predicted_d_value())

            self.predicted_beta_values = np.append(self.predicted_beta_values, self.beta)
            self.predicted_gamma_values = np.append(self.predicted_gamma_values, self.gamma)
            self.predicted_mu_values = np.append(self.predicted_mu_values, self.mu)



    def pred(self, nb_of_days = 60):
        for i in range(nb_of_days):
            self.ukf.predict(model_self=self)

            self.x = self.ukf.x[:3]
            self.beta = self.ukf.x[3]
            self.gamma = self.ukf.x[4]
            self.mu = self.ukf.x[5]

            self.ukf.update(np.array([self.predicted_i_value(), self.predicted_r_value(), self.predicted_d_value()]))
        
            self.data_s_values = np.append(self.data_s_values, self.s_at_day(i))
            self.data_i_values = np.append(self.data_i_values, self.i_at_day(i))
            self.data_r_values = np.append(self.data_r_values, self.r_at_day(i))
            self.data_d_values = np.append(self.data_d_values, self.d_at_day(i))

            self.predicted_s_values = np.append(self.predicted_s_values, self.predicted_s_value())
            self.predicted_i_values = np.append(self.predicted_i_values, self.predicted_i_value())
            self.predicted_r_values = np.append(self.predicted_r_values, self.predicted_r_value())
            self.predicted_d_values = np.append(self.predicted_d_values, self.predicted_d_value())

            self.predicted_beta_values = np.append(self.predicted_beta_values, self.beta)
            self.predicted_gamma_values = np.append(self.predicted_gamma_values, self.gamma)
            self.predicted_mu_values = np.append(self.predicted_mu_values, self.mu)

    def plot(self):
        days = range(self.predicted_s_values.size)
        figure, axes = plt.subplots(5, 1, figsize=(11,17))

        #Plot s
        s_plot = axes[0]
        s_plot.plot(days, self.predicted_s_values[:days], "#0072bd", label = "Predicted S")
        s_plot.bar(days, self.data_s_values, color = "#0072bd", alpha = 0.3)
        s_plot.legend(loc='best')

        #Plot I, R
        ir_plot = axes[1]
        ir_plot.plot(days, self.predicted_i_values[:days], "#d95319", label ="Predicted I")
        ir_plot.plot(days, self.predicted_r_values[:days], "#edb120", label ="Predicted R")


        ir_plot.bar(days, self.data_i_values, color= "#d95319",alpha = 0.3)
        ir_plot.bar(days, self.data_r_values, color ="#edb120",alpha = 0.3)
        ir_plot.legend(loc='best')

        #Plot D
        d_plot = axes[2]
        d_plot.plot(days, self.predicted_d_values[:days], "#7e2f8e", label ="Predicted D")
        d_plot.bar(days, self.data_d_values, color ="#470854", alpha = 0.3)
        d_plot.legend(loc='best')

        b_plot = axes[3]
        b_plot.plot(days, self.predicted_beta_values[:days], "#77ac30", label = "Beta")
        b_plot.legend(loc='best')

        bgm_plot = axes[4]
        bgm_plot.plot(days, self.predicted_gamma_values[:days], "#4dbeee", label = "Gamma")
        bgm_plot.plot(days, self.predicted_mu_values[:days], "#a2142f", label = "Mu")
        bgm_plot.legend(loc='best')

        plt.xlabel('time (day)')
        plt.show()

    def plot_ir(self,figsize = (13,9)):
        plt.rcParams["figure.figsize"] = figsize
        maxDays = self.data[:,1].shape[0]
        days = range(self.predicted_i_values.size)
        print("Current day = " + str(maxDays))


        plt.plot(days, self.predicted_i_values, "#0072bd", label = "Predicted I")
        #plt.bar(maxDays, self.data_i_values[:maxDays], color = "#0072bd", alpha = 0.3)

        plt.plot(days, self.predicted_r_values, "#edb120", label ="Predicted R")
        #plt.bar(maxDays, self.data_r_values[:maxDays], color ="#edb120",alpha = 0.3)

        plt.legend(loc='best')
        
        plt.show()

    def plot_d(self,figsize = (13,9)):

        plt.rcParams["figure.figsize"] = figsize
        maxDays = self.data[:,1].shape[0]
        days = range(self.predicted_i_values.size)
        print("Current day = " + str(maxDays))


        plt.plot(days, self.predicted_d_values, "#7e2f8e", label ="Predicted D")
        #plt.bar(days, self.data_d_values, color ="#470854", alpha = 0.3)

        plt.legend(loc='best')
        plt.show()