{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "blond-richmond",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "typical-cabin",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>R&amp;D Spend</th>\n",
       "      <th>Administration</th>\n",
       "      <th>Marketing Spend</th>\n",
       "      <th>State</th>\n",
       "      <th>Profit</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>165349.20</td>\n",
       "      <td>136897.80</td>\n",
       "      <td>471784.10</td>\n",
       "      <td>New York</td>\n",
       "      <td>192261.83</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>162597.70</td>\n",
       "      <td>151377.59</td>\n",
       "      <td>443898.53</td>\n",
       "      <td>California</td>\n",
       "      <td>191792.06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>153441.51</td>\n",
       "      <td>101145.55</td>\n",
       "      <td>407934.54</td>\n",
       "      <td>Florida</td>\n",
       "      <td>191050.39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>144372.41</td>\n",
       "      <td>118671.85</td>\n",
       "      <td>383199.62</td>\n",
       "      <td>New York</td>\n",
       "      <td>182901.99</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>142107.34</td>\n",
       "      <td>91391.77</td>\n",
       "      <td>366168.42</td>\n",
       "      <td>Florida</td>\n",
       "      <td>166187.94</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>131876.90</td>\n",
       "      <td>99814.71</td>\n",
       "      <td>362861.36</td>\n",
       "      <td>New York</td>\n",
       "      <td>156991.12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>134615.46</td>\n",
       "      <td>147198.87</td>\n",
       "      <td>127716.82</td>\n",
       "      <td>California</td>\n",
       "      <td>156122.51</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>130298.13</td>\n",
       "      <td>145530.06</td>\n",
       "      <td>323876.68</td>\n",
       "      <td>Florida</td>\n",
       "      <td>155752.60</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>120542.52</td>\n",
       "      <td>148718.95</td>\n",
       "      <td>311613.29</td>\n",
       "      <td>New York</td>\n",
       "      <td>152211.77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>123334.88</td>\n",
       "      <td>108679.17</td>\n",
       "      <td>304981.62</td>\n",
       "      <td>California</td>\n",
       "      <td>149759.96</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>101913.08</td>\n",
       "      <td>110594.11</td>\n",
       "      <td>229160.95</td>\n",
       "      <td>Florida</td>\n",
       "      <td>146121.95</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>100671.96</td>\n",
       "      <td>91790.61</td>\n",
       "      <td>249744.55</td>\n",
       "      <td>California</td>\n",
       "      <td>144259.40</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>93863.75</td>\n",
       "      <td>127320.38</td>\n",
       "      <td>249839.44</td>\n",
       "      <td>Florida</td>\n",
       "      <td>141585.52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>91992.39</td>\n",
       "      <td>135495.07</td>\n",
       "      <td>252664.93</td>\n",
       "      <td>California</td>\n",
       "      <td>134307.35</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>119943.24</td>\n",
       "      <td>156547.42</td>\n",
       "      <td>256512.92</td>\n",
       "      <td>Florida</td>\n",
       "      <td>132602.65</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>114523.61</td>\n",
       "      <td>122616.84</td>\n",
       "      <td>261776.23</td>\n",
       "      <td>New York</td>\n",
       "      <td>129917.04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>78013.11</td>\n",
       "      <td>121597.55</td>\n",
       "      <td>264346.06</td>\n",
       "      <td>California</td>\n",
       "      <td>126992.93</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>94657.16</td>\n",
       "      <td>145077.58</td>\n",
       "      <td>282574.31</td>\n",
       "      <td>New York</td>\n",
       "      <td>125370.37</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>91749.16</td>\n",
       "      <td>114175.79</td>\n",
       "      <td>294919.57</td>\n",
       "      <td>Florida</td>\n",
       "      <td>124266.90</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>86419.70</td>\n",
       "      <td>153514.11</td>\n",
       "      <td>0.00</td>\n",
       "      <td>New York</td>\n",
       "      <td>122776.86</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    R&D Spend  Administration  Marketing Spend       State     Profit\n",
       "0   165349.20       136897.80        471784.10    New York  192261.83\n",
       "1   162597.70       151377.59        443898.53  California  191792.06\n",
       "2   153441.51       101145.55        407934.54     Florida  191050.39\n",
       "3   144372.41       118671.85        383199.62    New York  182901.99\n",
       "4   142107.34        91391.77        366168.42     Florida  166187.94\n",
       "5   131876.90        99814.71        362861.36    New York  156991.12\n",
       "6   134615.46       147198.87        127716.82  California  156122.51\n",
       "7   130298.13       145530.06        323876.68     Florida  155752.60\n",
       "8   120542.52       148718.95        311613.29    New York  152211.77\n",
       "9   123334.88       108679.17        304981.62  California  149759.96\n",
       "10  101913.08       110594.11        229160.95     Florida  146121.95\n",
       "11  100671.96        91790.61        249744.55  California  144259.40\n",
       "12   93863.75       127320.38        249839.44     Florida  141585.52\n",
       "13   91992.39       135495.07        252664.93  California  134307.35\n",
       "14  119943.24       156547.42        256512.92     Florida  132602.65\n",
       "15  114523.61       122616.84        261776.23    New York  129917.04\n",
       "16   78013.11       121597.55        264346.06  California  126992.93\n",
       "17   94657.16       145077.58        282574.31    New York  125370.37\n",
       "18   91749.16       114175.79        294919.57     Florida  124266.90\n",
       "19   86419.70       153514.11             0.00    New York  122776.86"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/Startups_Data.csv')\n",
    "df.head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "thick-surveillance",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>R&amp;D Spend</th>\n",
       "      <th>Administration</th>\n",
       "      <th>Marketing Spend</th>\n",
       "      <th>Profit</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>50.000000</td>\n",
       "      <td>50.000000</td>\n",
       "      <td>50.000000</td>\n",
       "      <td>50.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>73721.615600</td>\n",
       "      <td>121344.639600</td>\n",
       "      <td>211025.097800</td>\n",
       "      <td>112012.639200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>45902.256482</td>\n",
       "      <td>28017.802755</td>\n",
       "      <td>122290.310726</td>\n",
       "      <td>40306.180338</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>51283.140000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14681.400000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>39936.370000</td>\n",
       "      <td>103730.875000</td>\n",
       "      <td>129300.132500</td>\n",
       "      <td>90138.902500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>73051.080000</td>\n",
       "      <td>122699.795000</td>\n",
       "      <td>212716.240000</td>\n",
       "      <td>107978.190000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>101602.800000</td>\n",
       "      <td>144842.180000</td>\n",
       "      <td>299469.085000</td>\n",
       "      <td>139765.977500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>165349.200000</td>\n",
       "      <td>182645.560000</td>\n",
       "      <td>471784.100000</td>\n",
       "      <td>192261.830000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           R&D Spend  Administration  Marketing Spend         Profit\n",
       "count      50.000000       50.000000        50.000000      50.000000\n",
       "mean    73721.615600   121344.639600    211025.097800  112012.639200\n",
       "std     45902.256482    28017.802755    122290.310726   40306.180338\n",
       "min         0.000000    51283.140000         0.000000   14681.400000\n",
       "25%     39936.370000   103730.875000    129300.132500   90138.902500\n",
       "50%     73051.080000   122699.795000    212716.240000  107978.190000\n",
       "75%    101602.800000   144842.180000    299469.085000  139765.977500\n",
       "max    165349.200000   182645.560000    471784.100000  192261.830000"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "private-surge",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>R&amp;D Spend</th>\n",
       "      <th>Administration</th>\n",
       "      <th>Marketing Spend</th>\n",
       "      <th>State</th>\n",
       "      <th>Profit</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>165349.20</td>\n",
       "      <td>136897.80</td>\n",
       "      <td>471784.10</td>\n",
       "      <td>2</td>\n",
       "      <td>192261.83</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>162597.70</td>\n",
       "      <td>151377.59</td>\n",
       "      <td>443898.53</td>\n",
       "      <td>0</td>\n",
       "      <td>191792.06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>153441.51</td>\n",
       "      <td>101145.55</td>\n",
       "      <td>407934.54</td>\n",
       "      <td>1</td>\n",
       "      <td>191050.39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>144372.41</td>\n",
       "      <td>118671.85</td>\n",
       "      <td>383199.62</td>\n",
       "      <td>2</td>\n",
       "      <td>182901.99</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>142107.34</td>\n",
       "      <td>91391.77</td>\n",
       "      <td>366168.42</td>\n",
       "      <td>1</td>\n",
       "      <td>166187.94</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   R&D Spend  Administration  Marketing Spend  State     Profit\n",
       "0  165349.20       136897.80        471784.10      2  192261.83\n",
       "1  162597.70       151377.59        443898.53      0  191792.06\n",
       "2  153441.51       101145.55        407934.54      1  191050.39\n",
       "3  144372.41       118671.85        383199.62      2  182901.99\n",
       "4  142107.34        91391.77        366168.42      1  166187.94"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Redefine State's data into integer, based on the order of the states inputted\n",
    "#New York = 0\n",
    "#California = 1\n",
    "#FLorida = 2\n",
    "\n",
    "df[['State']] = df[['State']].apply(lambda col:pd.Categorical(col).codes)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "stuck-monitoring",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>R&amp;D Spend</th>\n",
       "      <th>Administration</th>\n",
       "      <th>Marketing Spend</th>\n",
       "      <th>State</th>\n",
       "      <th>Profit</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>72107.60</td>\n",
       "      <td>127864.55</td>\n",
       "      <td>353183.81</td>\n",
       "      <td>2</td>\n",
       "      <td>105008.31</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>46014.02</td>\n",
       "      <td>85047.44</td>\n",
       "      <td>205517.64</td>\n",
       "      <td>2</td>\n",
       "      <td>96479.51</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40</th>\n",
       "      <td>28754.33</td>\n",
       "      <td>118546.05</td>\n",
       "      <td>172795.67</td>\n",
       "      <td>0</td>\n",
       "      <td>78239.91</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>38</th>\n",
       "      <td>20229.59</td>\n",
       "      <td>65947.93</td>\n",
       "      <td>185265.10</td>\n",
       "      <td>2</td>\n",
       "      <td>81229.06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>153441.51</td>\n",
       "      <td>101145.55</td>\n",
       "      <td>407934.54</td>\n",
       "      <td>1</td>\n",
       "      <td>191050.39</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    R&D Spend  Administration  Marketing Spend  State     Profit\n",
       "27   72107.60       127864.55        353183.81      2  105008.31\n",
       "35   46014.02        85047.44        205517.64      2   96479.51\n",
       "40   28754.33       118546.05        172795.67      0   78239.91\n",
       "38   20229.59        65947.93        185265.10      2   81229.06\n",
       "2   153441.51       101145.55        407934.54      1  191050.39"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Split data into 80:20 ratio for training and testing\n",
    "train_df = df.sample(frac = 0.8, random_state = 1)\n",
    "test_df = df.drop(train_df.index)\n",
    "train_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "helpful-puppy",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>R&amp;D Spend</th>\n",
       "      <th>Administration</th>\n",
       "      <th>Marketing Spend</th>\n",
       "      <th>State</th>\n",
       "      <th>Profit</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>165349.20</td>\n",
       "      <td>136897.80</td>\n",
       "      <td>471784.10</td>\n",
       "      <td>2</td>\n",
       "      <td>192261.83</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>131876.90</td>\n",
       "      <td>99814.71</td>\n",
       "      <td>362861.36</td>\n",
       "      <td>2</td>\n",
       "      <td>156991.12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>120542.52</td>\n",
       "      <td>148718.95</td>\n",
       "      <td>311613.29</td>\n",
       "      <td>2</td>\n",
       "      <td>152211.77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>123334.88</td>\n",
       "      <td>108679.17</td>\n",
       "      <td>304981.62</td>\n",
       "      <td>0</td>\n",
       "      <td>149759.96</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>100671.96</td>\n",
       "      <td>91790.61</td>\n",
       "      <td>249744.55</td>\n",
       "      <td>0</td>\n",
       "      <td>144259.40</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    R&D Spend  Administration  Marketing Spend  State     Profit\n",
       "0   165349.20       136897.80        471784.10      2  192261.83\n",
       "5   131876.90        99814.71        362861.36      2  156991.12\n",
       "8   120542.52       148718.95        311613.29      2  152211.77\n",
       "9   123334.88       108679.17        304981.62      0  149759.96\n",
       "11  100671.96        91790.61        249744.55      0  144259.40"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "qualified-hampshire",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     192261.83\n",
       "5     156991.12\n",
       "8     152211.77\n",
       "9     149759.96\n",
       "11    144259.40\n",
       "12    141585.52\n",
       "15    129917.04\n",
       "16    126992.93\n",
       "37     89949.14\n",
       "43     69758.98\n",
       "Name: Profit, dtype: float64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df_target = train_df[\"Profit\"]\n",
    "train_df.pop(\"Profit\")\n",
    "\n",
    "test_df_target = test_df[\"Profit\"]\n",
    "test_df.pop(\"Profit\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "hindu-cotton",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>R&amp;D Spend</th>\n",
       "      <th>Administration</th>\n",
       "      <th>Marketing Spend</th>\n",
       "      <th>State</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>10.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>98775.161000</td>\n",
       "      <td>113610.145000</td>\n",
       "      <td>270951.024000</td>\n",
       "      <td>1.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>43764.622916</td>\n",
       "      <td>27634.280755</td>\n",
       "      <td>112300.019896</td>\n",
       "      <td>0.994429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>15505.730000</td>\n",
       "      <td>51283.140000</td>\n",
       "      <td>35534.170000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>81975.770000</td>\n",
       "      <td>102030.825000</td>\n",
       "      <td>249768.272500</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>107597.785000</td>\n",
       "      <td>122107.195000</td>\n",
       "      <td>263061.145000</td>\n",
       "      <td>1.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>122636.790000</td>\n",
       "      <td>127366.820000</td>\n",
       "      <td>309955.372500</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>165349.200000</td>\n",
       "      <td>148718.950000</td>\n",
       "      <td>471784.100000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           R&D Spend  Administration  Marketing Spend      State\n",
       "count      10.000000       10.000000        10.000000  10.000000\n",
       "mean    98775.161000   113610.145000    270951.024000   1.100000\n",
       "std     43764.622916    27634.280755    112300.019896   0.994429\n",
       "min     15505.730000    51283.140000     35534.170000   0.000000\n",
       "25%     81975.770000   102030.825000    249768.272500   0.000000\n",
       "50%    107597.785000   122107.195000    263061.145000   1.500000\n",
       "75%    122636.790000   127366.820000    309955.372500   2.000000\n",
       "max    165349.200000   148718.950000    471784.100000   2.000000"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ahead-leadership",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>R&amp;D Spend</th>\n",
       "      <th>Administration</th>\n",
       "      <th>Marketing Spend</th>\n",
       "      <th>State</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>40.000000</td>\n",
       "      <td>40.000000</td>\n",
       "      <td>40.000000</td>\n",
       "      <td>40.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>67458.229250</td>\n",
       "      <td>123278.263250</td>\n",
       "      <td>196043.616250</td>\n",
       "      <td>0.97500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>44767.135002</td>\n",
       "      <td>28122.536337</td>\n",
       "      <td>121359.867133</td>\n",
       "      <td>0.80024</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>51743.150000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>28731.687500</td>\n",
       "      <td>105077.645000</td>\n",
       "      <td>115395.745000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>65828.500000</td>\n",
       "      <td>123467.895000</td>\n",
       "      <td>193195.960000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>91809.967500</td>\n",
       "      <td>145947.262500</td>\n",
       "      <td>297501.962500</td>\n",
       "      <td>2.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>162597.700000</td>\n",
       "      <td>182645.560000</td>\n",
       "      <td>443898.530000</td>\n",
       "      <td>2.00000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           R&D Spend  Administration  Marketing Spend     State\n",
       "count      40.000000       40.000000        40.000000  40.00000\n",
       "mean    67458.229250   123278.263250    196043.616250   0.97500\n",
       "std     44767.135002    28122.536337    121359.867133   0.80024\n",
       "min         0.000000    51743.150000         0.000000   0.00000\n",
       "25%     28731.687500   105077.645000    115395.745000   0.00000\n",
       "50%     65828.500000   123467.895000    193195.960000   1.00000\n",
       "75%     91809.967500   145947.262500    297501.962500   2.00000\n",
       "max    162597.700000   182645.560000    443898.530000   2.00000"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "formal-black",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
