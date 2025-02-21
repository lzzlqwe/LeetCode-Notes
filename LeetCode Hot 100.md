# 哈希表

## [1.字母异位词分组](https://leetcode.cn/problems/group-anagrams/description/)
思路：创建一个HashMap，对每个字符串进行排序，排序后相同结果的归在同一个key下。  
代码：
```
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> mp = new HashMap<>();
        for(String str : strs){
            char[] carr = str.toCharArray();
            Arrays.sort(carr);
            String key = String.valueOf(carr);
            if(mp.containsKey(key))
                mp.get(key).add(str);
            else{
                List<String> list = new ArrayList<>();
                list.add(str);
                mp.put(key, list); 
            }
        }
        return new ArrayList<>(mp.values());
    }
}
```

## [2.最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/)
思路：nums用一个hash表来存放，遍历hash表，如果存在比他小1的值，则考虑比他小的值即可，跳过；如果不存在比他小1的值，则说明当前值可能是最长连续序列的起点。  
代码：
```
class Solution {
    public int longestConsecutive(int[] nums) {
        //nums用一个hash表来存放
        Set<Integer> set = new HashSet<>();
        for(int x : nums) 
            set.add(x);
        int maxLength = 0;
        //遍历hash表
        for(int x : set){
            if(set.contains(x-1)) //如果存在比他小1的值，则考虑比他小的值即可。
                continue;
            int t = x; //不存在比他小1的值，说明当前值可能是最长连续序列的起点
            while(set.contains(t))
                t++;
            maxLength = Math.max(maxLength, t - x); //更新答案
        }
        return maxLength;
    }
}
```

# 双指针

## [1.移动零](https://leetcode.cn/problems/move-zeroes/description/)
思路：使用双指针，左指针始终指向第一个0， 右指针指向left后的第一个非零元素（其中left到right-1都为0），进行元素交换后，双指针继续向后移动。  
代码：
```
class Solution {
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        int left = 0; //始终指向第一个0
        while(left < n && nums[left] != 0)
            left++;
        int right = left; //指向left后的非零元素
        while(right < n && nums[right] == 0)
            right++;
        while(right < n){
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            while(right < n && nums[right] == 0)
                right++;
            while(left < n && nums[left] != 0)
                left++;
        }
    }
}
```

## [2.盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/)
思路：定义两个指针left和right，left从最左边开始，right从最右边开始，每次移动最矮的那条边（因为移动最矮边才可能在中间找到最优解），并记录储存的最大水量。  
代码：
```
class Solution {
    public int maxArea(int[] height) {
        int ans = 0;
        int left = 0;
        int right = height.length - 1;
        while(left < right){
            int s = (right - left) * Math.min(height[left], height[right]);
            ans = Math.max(ans, s);
            if(height[left] < height[right])
                left++;
            else
                right--;
        }
        return ans;
    }
}
```

## [3.三数之和](https://leetcode.cn/problems/3sum/description/)
思路：先排序，然后定义三个指针i，left和right，外层循环指针i从0遍历到倒数第三个位置（留两个位置给相向指针left和right），指针left从i+1开始，指针right从最右边开始。（如果题目要求结果不能重复，则三个指针都只指向第一个重复元素，绕过后续重复元素）  
代码：
```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums); //排序
        List<List<Integer>> ans = new ArrayList<>();
        int n = nums.length;
        int i = 0;
        while(i < n - 2){
            int left = i + 1;
            int right = n - 1;
            while(left < right){
                if(nums[i] + nums[left] + nums[right] == 0){
                    ans.add(new ArrayList(Arrays.asList(nums[i], nums[left], nums[right])));
                    left++;
                    while(left < right && nums[left-1] == nums[left])//指向第一个重复的元素
                        left++;
                    right--;
                    while(left < right && nums[right] == nums[right+1])//指向第一个重复的元素
                        right--;
                }
                else if(nums[i] + nums[left] + nums[right] < 0)
                    left++;
                else
                    right--;
            }
            i++;
            while(i < n - 2 && nums[i-1] == nums[i]) //指向第一个重复的元素,绕过后续重复元素
                i++;
        }
        return ans;
    }
}
```

## [4.接雨水](https://leetcode.cn/problems/trapping-rain-water/description/)
思路：定义两个数组，前缀数组记录从左边到下标i的最大高度，后缀数组记录从右边到下标i的最大高度，有了这两个数组就可以计算各个下标i的盛水单位量。  
代码：
```
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        int[] pre = new int[n];
        int[] post = new int[n];
        //求前缀数组和后缀数组
        int max = 0;
        for(int i = 0; i < n; i++){
            max = Math.max(max, height[i]);
            pre[i] = max;
        }
        max = 0;
        for(int i = n-1; i >= 0; i--){
            max = Math.max(max, height[i]);
            post[i] = max;
        }
        //根据前缀数组和后缀数组求答案
        int ans = 0;
        for(int i = 0; i < n; i++){
            ans += Math.min(pre[i], post[i]) - height[i];
        }
        return ans;
    }
}
```

# 滑动窗口

### 模板代码
```
int start = 0;
int end = 0;
while(end < n){ //外层循环扩展右边界，内层循环扩展左边界
    xxx;  //当前考虑的元素
    while(start <= end && 条件){
        xxx
        start++; //扩展左边界
    }
    end++; //扩展右边界
}
```

### 滑动窗口使用前提：  
1. 连续子数组。
2. 有单调性。比如元素均为正数。

## [1.无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)
思路：套用滑动窗口模板  
代码：
```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        char[] strArr = s.toCharArray();
        int n = strArr.length;
        int left = 0;
        int right = 0;
        int maxLength = 0;
        while(right < n){
            char c = strArr[right];
            while(left <= right && set.contains(c)){
                set.remove(strArr[left]);
                left++;
            }
            set.add(c);
            maxLength = Math.max(maxLength, right - left + 1);
            right++;
        }
        return maxLength;
    }
}
```
