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

## [5.将三个组排序](https://leetcode.cn/problems/sorting-three-groups/description/)
思路：求出最长非递减子序列的长度len，用总长度减去len即可。  
代码：
```
class Solution {
    public int minimumOperations(List<Integer> nums) {
        int n = nums.size();
        int[] d = new int[n+1];
        int len = 1;
        d[len] = nums.get(0);
        for(int i = 1; i < n; i++)
        {
            if(nums.get(i) >= d[len])
            {
                len++;
                d[len] = nums.get(i);
            }
            else
            {
                int idx = lower_bound(d, 1, len, nums.get(i));
                while(d[idx] == nums.get(i)) //因为idx应该指向d数组中相同元素的最后 
                    idx++;
                d[idx] = nums.get(i);
            }
        }
        return n - len;
    }

    //二分查找
    public int lower_bound(int[] nums, int left, int right, int target){
        int l = left;
        int r = right;
        while(l <= r){
            int mid = l + (r - l) / 2;
            if(nums[mid] < target)
                l = mid + 1;
            else
                r = mid - 1;
        }
        return l;
    }
}
```

## [6.访问数组中的位置使分数最大](https://leetcode.cn/problems/visit-array-positions-to-maximize-score/description/)
思路：详见代码注解或者官方题解。  
代码：
```
class Solution {
    public long maxScore(int[] nums, int x) {
        int n = nums.length;
        long[] scores = new long[n]; //scores[i]表示最后移动到位置i的最大得分
        scores[0] = nums[0]; //一开始的分数为 nums[0] 
        //dp[0]表示之前位置中最后移动的元素为偶数时得分的最大值
        //dp[1]表示之前位置中最后移动的元素为奇数时得分的最大值
        long[] dp = new long[]{Integer.MIN_VALUE, Integer.MIN_VALUE};
        //初始化dp 
        dp[nums[0] % 2] = nums[0];
        long ans = scores[0]; //记录答案
        //依次遍历数组，计算最后移动到位置i的最大得分
        for(int i = 1; i < n; i++){
            if(nums[i] % 2 == 0) //当前位置为偶数
            {
                scores[i] = Math.max(dp[0] + nums[i], dp[1] + nums[i] - x);
                dp[0] = scores[i]; //更新之前位置中最后移动的元素为偶数时得分的最大值
            }
            else //当前位置为奇数
            {
                scores[i] = Math.max(dp[1] + nums[i], dp[0] + nums[i] - x);
                dp[1] = scores[i]; //更新之前位置中最后移动的元素为奇数时得分的最大值
            }
            ans = Math.max(ans, scores[i]);//更新答案
        }
        return ans;
    }
}
```
