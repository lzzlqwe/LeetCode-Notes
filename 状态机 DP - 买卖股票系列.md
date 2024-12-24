# 状态机 DP

## [1.买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/description/)
思路：dp[i][0]表示第i天后不持有股票的最大利润，dp[i][1]则表示持有股票。状态转移方程为：dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices); （保持不变或者出售股票->第i天后不持有股票） dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] - prices); （保持不变或者购买股票->第i天后持有股票）     
代码：
```
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n+1][2]; //dp[i][0]表示第i天后不持有股票的最大利润，dp[i][1]则表示持有股票
        dp[0][0] = 0;
        dp[0][1] = Integer.MIN_VALUE; //初始化
        //状态转移方程为
        //dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices); //保持不变或者出售股票->第i天后不持有股票
        //dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] - prices); //保持不变或者购买股票->第i天后持有股票
        for(int i = 1; i <= n; i++){
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices[i-1]);
            dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] - prices[i-1]);
        }
        return dp[n][0]; //第n天后的最大利润，肯定不持有股票
    }
}
```

## [2.买卖股票的最佳时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/)
思路：和上一题思路类似，但需要修改一下状态转移方程。 dp[i][1] = Math.max(dp[i-1][1], dp[i-2][0] - prices)（若买入股票，则考虑前两天未持股的状态！！！ ）   
代码：
```
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n+1][2]; //dp[i][0]表示第i天后不持有股票的最大利润，dp[i][1]则表示持有股票
        dp[0][0] = 0;
        dp[0][1] = Integer.MIN_VALUE; //初始化
        //状态转移方程为
        //dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices); 
        //dp[i][1] = Math.max(dp[i-1][1], dp[i-2][0] - prices); //若买入股票，则考虑前两天未持股的状态！！！
        for(int i = 1; i <= n; i++){
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices[i-1]);
            if(i == 1) dp[i][1] = - prices[i-1]; //直接计算持有股第一天结束后的最大利润
            else dp[i][1] = Math.max(dp[i-1][1], dp[i-2][0] - prices[i-1]);
        }
        return dp[n][0]; //第n天后的最大利润，肯定不持有股票
    }
}
```

## [3.买卖股票的最佳时机 IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/description/)
思路：dp[i][j][0]表示第i天结束，未持股，最多完成j笔交易情况下的最大利润。dp[i][j][1]则表示持股情况下。状态转移方程为 未持股：dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + price) 不动或者购入股票。持股：dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - price) 不动或者出售股票   
代码：
```
class Solution {
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        //dp[i][j][0]表示第i天结束，未持股，最多完成j笔交易情况下的最大利润。dp[i][j][1]表示持股
        int[][][] dp = new int[n+1][k+1][2];
        //初始化
        for(int j = 0; j <= k; j++) dp[0][j][0] = 0;
        for(int i = 0; i <= n; i++) dp[i][0][0] = 0;
        for(int j = 0; j <= k; j++) dp[0][j][1] = Integer.MIN_VALUE;
        for(int i = 0; i <= n; i++) dp[i][0][1] = Integer.MIN_VALUE;
        //状态转移方程
        //未持股 dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + price) 不动或者购入股票
        //持股 dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - price) 不动或者出售股票
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= k; j++){
                dp[i][j][0] = Math.max(dp[i-1][j][0], dp[i-1][j][1] + prices[i-1]);
                dp[i][j][1] = Math.max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i-1]);
            }
        }
        return dp[n][k][0];
    }
}
```

## [4.买卖股票的最佳时机含手续费](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/description/)
思路：和第一一题思路类似，但需要修改一下状态转移方程。 dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices - fee); 保持不变或者出售股票（需要考虑手续费）。此外，在初始化时应该注意防止意出。  
代码：
```
class Solution {
    public int maxProfit(int[] prices, int fee) {
        int n = prices.length;
        int[][] dp = new int[n+1][2]; //dp[i][0]表示第i天后不持有股票的最大利润，dp[i][1]则表示持有股票
        dp[0][0] = 0;
        dp[0][1] = Integer.MIN_VALUE / 2; //初始化，防止溢出！！！
        //状态转移方程为
        //dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices - fee); //保持不变或者出售股票
        //dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] - prices); //保持不变或者购买股票
        for(int i = 1; i <= n; i++){
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices[i-1] - fee);
            dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] - prices[i-1]);
        }
        return dp[n][0]; //第n天后的最大利润，肯定不持有股票
    }
}
```
