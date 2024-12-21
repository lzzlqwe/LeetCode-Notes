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
        dp[0][1] = -Integer.MIN_VALUE; //初始化
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
