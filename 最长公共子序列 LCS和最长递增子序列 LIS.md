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

## [2.编辑距离](https://leetcode.cn/problems/edit-distance/description/)
思路：dp[i][j]表示word1前i个字符转换为word2的前j个字符的最少操作数。状态转移方程如下：   
![image](https://github.com/user-attachments/assets/102dd6bd-f89d-45b7-bfe9-645ca554dae9)

代码：
```
class Solution {
    public int minDistance(String word1, String word2) {
        char[] str1 = word1.toCharArray();
        char[] str2 = word2.toCharArray();
        int n = str1.length;
        int m = str2.length;
        int[][] dp = new int[n+1][m+1]; //dp[i][j]表示word1前i个字符转换为word2的前j个字符的最少操作数 
        for(int i = 0; i <= n; i++) dp[i][0] = i; //根据实际意义初始化！！！
        for(int j = 0; j <= m; j++) dp[0][j] = j; //根据实际意义初始化！！！
        for(int i = 1; i <= n; i++)
        {
            for(int j = 1; j <= m; j++)
            {
                //状态转移方程
                if(str1[i-1] == str2[j-1])
                    dp[i][j] = dp[i-1][j-1];
                else
                    dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])+1;
            }
        }
        return dp[n][m];
    }

    public int min(int a, int b, int c)
    {
        return Math.min(Math.min(a, b), c);
    }
}
```

# 最长递增子序列 LIS

## [1.最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/description/)
思路一：dp[i]表示以nums[i]元素结尾的最长递增子序列长度。状态转移方程为dp[i] = max{dp[j]}+1，其中j<i且nums[j] < nums[i]。      
代码：
```
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n]; //dp[i]表示以nums[i]元素结尾的最长递增子序列长度。
        //状态转移方程为dp[i] = max{dp[j]}+1，其中j<i且nums[j] < nums[i]
        int ans = 0;
        for(int i = 0; i < n; i++)
        {
            int max = 0;
            for(int j = 0; j < i; j++)
            {
                if(nums[j] < nums[i] && dp[j] > max)
                    max = dp[j];
            }
            dp[i] = max + 1;
            if(dp[i] > ans) ans = dp[i];
        }
        return ans;
    }
}
```

思路二：贪心 + 二分查找。考虑一个简单的贪心，如果我们要使上升子序列尽可能的长，则我们需要让序列上升得尽可能慢，因此我们希望每次在上升子序列最后加上的那个数尽可能的小。基于上面的贪心思路，我们维护一个数组 d[i] ，表示长度为 i 的最长上升子序列的末尾元素的最小值，用 len 记录目前最长上升子序列的长度，起始时 len 为 1，d[1]=nums[0]。依次遍历数组 nums 中的每个元素，并更新数组 d 和 len 的值。如果 nums[i]>d[len] 则更新 len=len+1，并将nums[i]添加到d数组末尾。否则在 d[1…len]（二分查找闭区间写法，下标从1到len）中找到第一个大于等于nums[i]的下标k，将d[k]覆盖为nums[i]。最终返回len即为结果。

代码：
```
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] d = new int[n+1]; //d[i]表示长度为i的严格递增子序列的末尾最小元素
        int len = 1; //记录目前最长上升子序列的长度
        d[len] = nums[0]; //d数组可证明是单调递增的
        for(int j = 1; j < n; j++)
        {
            if(nums[j] > d[len])
            {
                len++;
                d[len] = nums[j];
            }
            else
            {
                int index = lower_bound(d, 1, len, nums[j]);
                d[index] = nums[j];
            }
        }
        return len;
    }
    
    //二分查找
    public int lower_bound(int[] d, int left, int right, int target)
    {
        int l = left;
        int r = right;
        while(l <= r)
        {
            int mid = (r - l) / 2 + l;
            if(d[mid] < target)
                l = mid + 1;
            else
                r = mid - 1;
        }
        return l;
    }
}
```

## [2.得到山形数组的最少删除次数](https://leetcode.cn/problems/minimum-number-of-removals-to-make-mountain-array/description/)
思路：按照灵神的思路。参考上题思路二来做。详见代码注释。     
代码：
```
class Solution {
    public int minimumMountainRemovals(int[] nums) {
        int n = nums.length;
        int[] up = new int[n]; //up[i]表示以nums[i]结尾(不一定包括)的最长严格递增子序列长度
        int[] low = new int[n]; //low[i]表示以nums[i]开头(不一定包括)的最长严格递减子序列长度
        int[] d = new int[n+1]; //d[i]表示当前长度为i的最长严格递增子序列的结尾最小值
        int len = 1; //当前最长严格递增子序列长度
        //计算up数组
        d[len] = nums[0];
        up[0] = 1;
        for(int i = 1; i < n; i++)
        {
            if(nums[i] > d[len])
            {
                len++;
                d[len] = nums[i];
            }
            else
            {
                int index = lower_bound(d, 1, len, nums[i]);
                d[index] = nums[i];
            }
            up[i] = len;
        }
        //计算low数组
        len = 1; //d[i]表示当前长度为i的最长严格递减子序列的开头最小值
        Arrays.fill(d, 0);
        d[len] = nums[n-1];
        low[n-1] = 1;
        for(int i = n-2; i >= 0; i--)
        {
            if(nums[i] > d[len])
            {
                len++;
                d[len] = nums[i];
            }
            else
            {
                int index = lower_bound(d, 1, len, nums[i]);;
                d[index] = nums[i];
            }
            low[i] = len;
        }
        //计算山形数组的最长长度
        int max = 0;
        for(int i = 0; i < n; i++)  //题目要求，峰顶左右必须有值！！！
            if(up[i] > 1 && low[i] > 1)
                max = Math.max(max, up[i]+low[i]-1);
        return n - max; //总长度-山形数组的最长长度=最少删除次数
    }

    //二分查找
    public int lower_bound(int[] d, int left, int right, int target)
    {
        int l = left;
        int r = right;
        while(l <= r)
        {
            int mid = (r - l) / 2 + l;
            if(d[mid] < target)
                l = mid + 1;
            else
                r = mid - 1;
        }
        return l;
    }
}
```
