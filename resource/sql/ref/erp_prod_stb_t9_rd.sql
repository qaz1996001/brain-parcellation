CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `erp_prod_stb_t9_rd`
--

DROP TABLE IF EXISTS `erp_prod_stb_t9_rd`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `erp_prod_stb_t9_rd` (
  `pri` int(11) NOT NULL DEFAULT 99 COMMENT 'priority: 1~99, but part different',
  `status1` int(11) NOT NULL DEFAULT 1 COMMENT 'complete: 0: wait stb 1:ongoing 9: complete\r\n\r\nstatus is mysql keyword, so rename to status1 -- 2020/01/02 lkchena',
  `tool_grp_id` varchar(24) NOT NULL,
  `tool_id` varchar(12) NOT NULL COMMENT 'length extend to 12 for sintering "before/after" omi -- 2020/03/18 lkchena',
  `part_id` varchar(24) NOT NULL,
  `part_raw` varchar(24) DEFAULT 'NULL',
  `part_name` varchar(24) DEFAULT 'NULL' COMMENT '品名 -- 2019/12/31 lkchena',
  `part_erp_no` varchar(24) DEFAULT 'NULL' COMMENT '料號 -- 2019/12/31 lkchena',
  `mold_id` varchar(16) NOT NULL,
  `powder_type` varchar(16) DEFAULT 'NULL',
  `powder_erp_no` varchar(24) DEFAULT 'NULL' COMMENT '粉料料號 -- 2019/12/31 lkchena',
  `cnt_plan` int(11) NOT NULL DEFAULT 0,
  `cnt_act` int(11) NOT NULL DEFAULT 0,
  `date_due` varchar(10) NOT NULL COMMENT 'must be key, cause maybe same product get many order -- 2019/12/10 lkchena\r\nyyyy/mm/dd hh:mm:ss',
  `date_stb` varchar(10) DEFAULT 'NULL' COMMENT 'yyyy/mm/dd hh:mm:ss',
  `date_comp` varchar(10) DEFAULT 'NULL' COMMENT 'yyyy/mm/dd hh:mm:ss',
  `mfg_no` varchar(16) DEFAULT 'NULL' COMMENT '??? why has mfg_no --> Trinity has the mfg_no, why ??? 2019/12/31 lkchena',
  `order_no` varchar(16) DEFAULT 'NULL',
  `rd_flag` int(11) DEFAULT 0 COMMENT '0: prod 1: rd -- 2020/01/28 lkchena',
  `cust_id` varchar(12) DEFAULT 'NULL',
  `weight` float DEFAULT 0,
  `unit` int(11) DEFAULT 1 COMMENT 'weight unit: 1: kg 2: ton (tonnage) 2019/12/31 lkchena',
  `owner_id` varchar(16) DEFAULT 'NULL' COMMENT 'order owner, when something wrong, mfg can find sponsor -- 2019/12/31 lkchena',
  `owner_name` varchar(24) DEFAULT 'NULL',
  `note` varchar(128) DEFAULT 'NULL',
  `rec_user` varchar(16) DEFAULT 'NULL',
  `rec_time` varchar(19) DEFAULT 'NULL',
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`tool_grp_id`,`tool_id`,`part_id`,`mold_id`,`date_due`),
  KEY `idx1` (`tool_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=2730 ROW_FORMAT=DYNAMIC COMMENT='for excel import, temp table, will use stored procedure to calculating final data -- 2020/01/01 lkchena';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `erp_prod_stb_t9_rd`
--

LOCK TABLES `erp_prod_stb_t9_rd` WRITE;
/*!40000 ALTER TABLE `erp_prod_stb_t9_rd` DISABLE KEYS */;
INSERT INTO `erp_prod_stb_t9_rd` VALUES (70,0,'FORMING-150T','AF15T01','T-ADB58F-1',NULL,NULL,NULL,'T-ADB58F-1','DR15M',NULL,926,0,'2019/02/05','2019/12/25',NULL,'T-MFG_NO-001',NULL,1,NULL,0,1,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-02-16 01:13:49.485090'),(33,0,'FORMING-150T','AF15T01','T-ADB60F',NULL,NULL,NULL,'T-ADB60F','C15M',NULL,778,0,'2019/02/05','2019/12/24',NULL,'T-MFG_NO-003',NULL,1,NULL,0,1,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-02-16 01:13:49.485090'),(40,0,'FORMING-150T','AF15T01','T-BBW131F',NULL,NULL,NULL,'T-BBW131F','C15M',NULL,28847,0,'2019/01/21','2019/12/23',NULL,'T-MFG_NO-004',NULL,1,NULL,0,1,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-02-16 01:13:49.485090'),(70,0,'FORMING-150T','AF15Y09','T-ADB58F-1',NULL,NULL,NULL,'T-ADB58F-1','DR15M',NULL,926,0,'2019/02/05','2019/12/25',NULL,'T-MFG_NO-001',NULL,1,NULL,0,1,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-02-16 01:13:49.485090'),(33,0,'FORMING-150T','AF15Y09','T-ADB60F',NULL,NULL,NULL,'T-ADB60F','C15M',NULL,778,0,'2019/02/05','2019/12/24',NULL,'T-MFG_NO-003',NULL,1,NULL,0,1,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-02-16 01:13:49.485090'),(40,0,'FORMING-150T','AF15Y09','T-BBW131F',NULL,NULL,NULL,'T-BBW131F','C15M',NULL,28847,0,'2019/01/21','2019/12/23',NULL,'T-MFG_NO-004',NULL,1,NULL,0,1,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-02-16 01:13:49.485090'),(70,0,'FORMING-350T','AF35Y01','T-ADB58F-1',NULL,NULL,NULL,'T-ADB58F-1','DR15M',NULL,926,0,'2019/02/05','2019/12/25',NULL,'T-MFG_NO-001',NULL,1,NULL,0,1,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-02-16 01:13:49.485090'),(33,0,'FORMING-350T','AF35Y01','T-ADB60F',NULL,NULL,NULL,'T-ADB60F','C15M',NULL,778,0,'2019/02/05','2019/12/24',NULL,'T-MFG_NO-003',NULL,1,NULL,0,1,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-02-16 01:13:49.485090'),(40,0,'FORMING-350T','AF35Y01','T-BBW131F',NULL,NULL,NULL,'T-BBW131F','C15M',NULL,28847,0,'2019/01/21','2019/12/23',NULL,'T-MFG_NO-004',NULL,1,NULL,0,1,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-02-16 01:13:49.485090');
/*!40000 ALTER TABLE `erp_prod_stb_t9_rd` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:32
