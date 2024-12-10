# 最长公共子序列 LCS

## [1.最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/description/)
思路：dp[i][j]表示text1前i和text2前j个字符的LCS长度。状态转移方程为，如果第i和第j个字符一样，则dp[i][j] = dp[i-1][j-1] + 1，否则dp[i][j] = max(dp[i-1][j], dp[i][j-1])。     
代码：
```
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        char[] str1 = text1.toCharArray();
        char[] str2 = text2.toCharArray();
        int n = str1.length;
        int m = str2.length;
        int[][] dp = new int[n+1][m+1]; //dp[i][j]表示text1前i和text2前j个字符的LCS长度
        for(int i = 1; i <= n; i++)
        {
            for(int j = 1; j <= m; j++)
            {
                //状态转移方程
                if(str1[i-1] == str2[j-1])
                    dp[i][j] = dp[i-1][j-1] + 1;
                else
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[n][m];
    }
}
```
